"""
Step 3 - Pretraining Masked Language Model pada korpus artikel CLICK-ID.

Jalankan dari root repo (setelah python -m src.data.prepare):
    python -m src.pretrain                                   # konfigurasi default config.PretrainConfig
    python -m src.pretrain --lr 1e-3 --batch-size 64 --run-name lr1e-3_bs64
    python -m src.pretrain --resume                          # lanjutkan dari checkpoints/pretrain/<run>/last.pt
    python -m src.pretrain --epochs 1 --max-steps 50         # uji cepat

Training: AdamW (tanpa weight decay untuk bias & LayerNorm), linear warmup lalu linear decay,
gradient clipping, mixed precision fp16 (hanya di GPU), dynamic masking (src/collator.py).

Output per run (<run> = --run-name, default "main"):
    checkpoints/pretrain/<run>/last.pt     model + optimizer + scheduler (untuk --resume), disimpan
                                           tiap pretrain_cfg.save_every step dan tiap akhir epoch
    checkpoints/pretrain/<run>/best.pt     model dengan loss validasi terendah (dipakai Step 4)
    reports/pretrain/<run>/history.json    train loss per log, metrik validasi per epoch, contoh fill-mask
    reports/pretrain/<run>/loss.png        grafik loss dan akurasi top-1/top-5
"""

import argparse
import json
import math
import time
from dataclasses import asdict

import torch
from tqdm import tqdm

from config import SEED, model_cfg, paths, pretrain_cfg
from src.collator import mask_tokens
from src.data.dataset import build_pretrain_loader
from src.data.tokenizer import load_tokenizer
from src.fill_mask import EXAMPLES, fill_mask, format_result
from src.model import IGNORE_INDEX, IndoMiniBERT
from src.plots import plot_pretrain


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # bias dan LayerNorm (parameter 1 dimensi) tidak diberi weight decay
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def linear_warmup_decay(warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return factor


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=None):
    """Loss, perplexity, akurasi top-1/top-5 pada token ter-mask. Masking validasi memakai seed tetap."""
    model.eval()
    generator = torch.Generator().manual_seed(SEED)
    total_loss, n_tokens, top1, top5 = 0.0, 0, 0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        inputs, labels = mask_tokens(batch["input_ids"], generator=generator)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(inputs.to(device), batch["attention_mask"].to(device), labels=labels.to(device))
        n = out["target"].numel()
        total_loss += out["loss"].item() * n
        n_tokens += n
        top = out["logits"].topk(5, dim=-1).indices
        hit = top == out["target"][:, None]
        top1 += hit[:, 0].sum().item()
        top5 += hit.any(-1).sum().item()
    model.train()
    loss = total_loss / max(n_tokens, 1)
    return {"loss": round(loss, 4), "ppl": round(math.exp(min(loss, 50)), 2),
            "top1": round(top1 / max(n_tokens, 1), 4), "top5": round(top5 / max(n_tokens, 1), 4),
            "masked_tokens": n_tokens}


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def train(lr=pretrain_cfg.lr, batch_size=pretrain_cfg.batch_size, epochs=pretrain_cfg.epochs, run_name="main",
          max_steps=None, max_eval_batches=None, resume=False, num_workers=pretrain_cfg.num_workers, device=None):
    """max_steps: batas step per epoch (untuk uji cepat / sweep). Kembalikan history."""
    torch.manual_seed(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = pretrain_cfg.use_amp and device == "cuda"
    ckpt_dir = paths.checkpoints_dir / "pretrain" / run_name
    report_dir = paths.reports_dir / "pretrain" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_pretrain_loader("train", batch_size=batch_size, num_workers=num_workers)
    val_loader = build_pretrain_loader("val", batch_size=batch_size, num_workers=num_workers)
    steps_per_epoch = min(len(train_loader), max_steps or len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * pretrain_cfg.warmup_ratio)

    model = IndoMiniBERT(model_cfg).to(device)
    optimizer = build_optimizer(model, lr, pretrain_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tokenizer = load_tokenizer()

    history = {
        "run_name": run_name,
        "config": {"lr": lr, "batch_size": batch_size, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
                   "total_steps": total_steps, "warmup_steps": warmup_steps, "device": device, "amp": use_amp,
                   "train_blocks": len(train_loader.dataset), "val_blocks": len(val_loader.dataset),
                   "pretrain_cfg": asdict(pretrain_cfg), "model_cfg": asdict(model_cfg),
                   "parameters": model.num_parameters()["total"]},
        "train": [], "val": [], "fill_mask": [],
    }
    global_step, best_val = 0, float("inf")

    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        global_step, best_val, history = ckpt["global_step"], ckpt["best_val"], ckpt["history"]
        print(f"Lanjut dari {last_path} (step {global_step})")

    def save_last():
        torch.save({"model": model.state_dict(), "model_cfg": asdict(model_cfg), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "global_step": global_step,
                    "best_val": best_val, "history": history}, last_path)

    print(f"[{run_name}] device={device} amp={use_amp} lr={lr} batch={batch_size} "
          f"step/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")
    model.train()
    start_epoch = global_step // steps_per_epoch
    for epoch in range(start_epoch, epochs):
        done_in_epoch = global_step - epoch * steps_per_epoch
        running, running_n, t0 = 0.0, 0, time.time()
        bar = tqdm(train_loader, total=steps_per_epoch - done_in_epoch, desc=f"epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(bar):
            # setelah --resume, sisa step epoch ini saja yang dijalankan (urutan batch diacak ulang)
            if i >= steps_per_epoch - done_in_epoch:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            inputs, labels = mask_tokens(input_ids)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(inputs, attention_mask, labels=labels)["loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running += loss.item()
            running_n += 1
            if global_step % pretrain_cfg.log_every == 0:
                avg = running / running_n
                history["train"].append({"step": global_step, "epoch": epoch + 1, "loss": round(avg, 4),
                                         "lr": scheduler.get_last_lr()[0]})
                bar.set_postfix(loss=f"{avg:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                running, running_n = 0.0, 0
            if global_step % pretrain_cfg.save_every == 0:
                save_last()

        val = evaluate(model, val_loader, device, use_amp, max_eval_batches)
        val.update({"epoch": epoch + 1, "step": global_step, "minutes": round((time.time() - t0) / 60, 2)})
        history["val"].append(val)
        examples = {text: fill_mask(model, tokenizer, text) for text in EXAMPLES}
        history["fill_mask"].append({"epoch": epoch + 1, "examples": examples})
        print(f"  val epoch {epoch + 1}: loss {val['loss']} | ppl {val['ppl']} | "
              f"top1 {val['top1']:.3f} | top5 {val['top5']:.3f}")
        for text, result in list(examples.items())[:2]:
            print("  " + format_result(text, result))

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"model": model.state_dict(), "model_cfg": asdict(model_cfg), "epoch": epoch + 1,
                        "step": global_step, "val": val, "run_name": run_name}, best_path)
            print(f"  checkpoint terbaik disimpan: {best_path}")
        history["best_val_loss"] = best_val
        save_last()
        save_json(history, report_dir / "history.json")
        plot_pretrain(history, report_dir / "loss.png")

    print(f"Selesai. Loss validasi terbaik {best_val}. Grafik: {report_dir / 'loss.png'}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=pretrain_cfg.lr)
    parser.add_argument("--batch-size", type=int, default=pretrain_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=pretrain_cfg.epochs)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--max-steps", type=int, default=None, help="batas step per epoch (uji cepat)")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=pretrain_cfg.num_workers)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    a = parser.parse_args()
    train(a.lr, a.batch_size, a.epochs, a.run_name, a.max_steps, a.max_eval_batches, a.resume, a.num_workers, a.device)


if __name__ == "__main__":
    main()
