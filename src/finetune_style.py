"""
Step 4 - Fine-tuning MLM Bergaya (Style-conditioned MLM).

Melanjutkan training MLM dari checkpoint pretraining (Step 3) pada judul berlabel.
Token gaya [CB] atau [NCB] ditempatkan di posisi 1 (setelah [CLS]) dan tidak pernah di-mask.

Cara jalankan dari root repo:
    python -m src.finetune_style
    python -m src.finetune_style --epochs 20 --batch-size 64 --device cuda

Cara resume jika terputus:
    python -m src.finetune_style --resume

Output:
    checkpoints/style_mlm.pt          - checkpoint terbaik (val loss terendah)
    reports/style_mlm_metrics.json    - history loss per epoch
    reports/style_mlm_loss.png        - grafik train/val loss
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import SEED, paths, pretrain_cfg, style_cfg
from src.collator import DynamicMaskingCollator
from src.data.dataset import build_title_loader
from src.model import BertForMaskedLM


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_linear_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.06):
    """Linear warmup lalu linear decay ke 0, persis seperti Step 3."""
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = total_steps - warmup_steps
        return max(0.0, float(total_steps - step) / max(remaining, 1))

    return LambdaLR(optimizer, _lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    """Satu epoch training atau validasi (optimizer=None → mode evaluasi)."""
    training = optimizer is not None
    model.train(training)
    criterion = CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        prediction_mask = labels.ne(-100)

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        use_amp = scaler is not None and scaler.is_enabled() and device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask, prediction_mask=prediction_mask)
                targets = labels[prediction_mask]
                loss = criterion(logits, targets)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        token_count = int(prediction_mask.sum().item())
        total_loss += loss.item() * token_count
        total_tokens += token_count
        correct += int(logits.argmax(dim=-1).eq(targets).sum().item()) if token_count > 0 else 0

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity, "masked_tokens": total_tokens}


def save_loss_plot(history, output_path: Path) -> None:
    """Simpan grafik train vs val loss sebagai PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, marker="o", label="train")
        ax.plot(epochs, val_losses, marker="s", label="validasi")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MLM Cross-Entropy Loss")
        ax.set_title("IndoMiniBERT – Style MLM Fine-Tuning Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib tidak tersedia, grafik tidak disimpan.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Fine-tune style-conditioned MLM pada judul berlabel CLICK-ID"
    )
    parser.add_argument("--epochs", type=int, default=style_cfg.epochs,
                        help=f"Jumlah epoch (default: {style_cfg.epochs})")
    parser.add_argument("--batch-size", type=int, default=style_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=style_cfg.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--init-checkpoint",
        default=str(paths.checkpoints_dir / "pretrain_mlm.pt"),
        help="Checkpoint Step 3 yang dipakai sebagai titik awal (wajib ada)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Lanjutkan training dari style_mlm_latest.pt jika ada"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia pada environment ini")

    init_path = Path(args.init_checkpoint)
    best_path = paths.checkpoints_dir / "style_mlm.pt"
    latest_path = paths.checkpoints_dir / "style_mlm_latest.pt"
    report_path = paths.reports_dir / "style_mlm_metrics.json"
    plot_path = paths.reports_dir / "style_mlm_loss.png"

    if not init_path.exists():
        raise FileNotFoundError(
            f"Checkpoint pretraining Step 3 tidak ditemukan di: {init_path}\n"
            "Penyebab: File checkpoint tidak ikut terbawa saat git clone (ada di .gitignore).\n"
            "Solusi:\n"
            "  1. Upload file 'checkpoints/pretrain_mlm.pt' (~29.5 MB) dari laptop ke folder checkpoints/ di Colab.\n"
            "  2. Atau jalankan pretraining terlebih dahulu: python -m src.pretrain --epochs 5 --device cuda\n"
        )

    set_seed()

    # Collator: protected_positions=(0,1) → [CLS] dan token gaya tidak pernah di-mask
    collator = DynamicMaskingCollator(
        mask_prob=pretrain_cfg.mask_prob,
        mask_replace_prob=pretrain_cfg.mask_replace_prob,
        random_replace_prob=pretrain_cfg.random_replace_prob,
        protected_positions=(0, 1),
    )
    train_loader = build_title_loader(
        "train", batch_size=args.batch_size, with_style=True, shuffle=True, collate_fn=collator
    )
    val_loader = build_title_loader(
        "val", batch_size=args.batch_size, with_style=True, shuffle=False, collate_fn=collator
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    model = BertForMaskedLM().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and pretrain_cfg.use_amp))
    history = []
    best_val_loss = float("inf")
    start_epoch = 1

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        print(f"Resume dari epoch {start_epoch}", flush=True)
    else:
        # Mulai dari checkpoint pretraining Step 3
        ckpt = torch.load(init_path, map_location=device)
        saved_state = ckpt["model_state"]

        # Deteksi format checkpoint: apakah menggunakan nama lama (query/key/value/output)
        # atau nama baru (q_proj/k_proj/v_proj/out_proj) dan apakah pakai mlm_head atau mlm_transform.
        # Checkpoint Step 3 mungkin berbeda versi dengan model saat ini, jadi lakukan remapping.
        remap = {
            "attention.query.": "attention.q_proj.",
            "attention.key.":   "attention.k_proj.",
            "attention.value.": "attention.v_proj.",
            "attention.output.": "attention.out_proj.",
            "intermediate.":    "ffn.net.0.",   # Linear(hidden→ffn) ada di ffn.net[0]
        }
        # Petakan keys FFN output: "output.weight" pada BertLayer (bukan attention.output)
        # Dalam arsitektur lama: layer.output.weight → ffn.net[3], layer.output_norm → ffn_norm
        remapped_state = {}
        for key, val in saved_state.items():
            new_key = key
            for old_suffix, new_suffix in remap.items():
                if old_suffix in new_key:
                    new_key = new_key.replace(old_suffix, new_suffix)
                    break
            # FFN output linear: bert.layers.N.output.weight → bert.layers.N.ffn.net.3.weight
            # tetapi juga harus menghindari attention.output yang sudah diremap
            if ".output.weight" in new_key and "attention" not in new_key:
                new_key = new_key.replace(".output.weight", ".ffn.net.3.weight")
            if ".output.bias" in new_key and "attention" not in new_key:
                new_key = new_key.replace(".output.bias", ".ffn.net.3.bias")
            if ".output_norm." in new_key:
                new_key = new_key.replace(".output_norm.", ".ffn_norm.")
            # mlm_transform + mlm_norm + mlm_bias → mlm_head (skip, tidak kompatibel langsung)
            # Hanya muat bobot encoder (bert.*), lewati mlm head lama
            remapped_state[new_key] = val

        # Filter hanya bobot encoder; mlm_head akan tetap dengan inisialisasi acak
        current_state = model.state_dict()
        encoder_keys = {k for k in current_state if k.startswith("bert.")}
        load_state = {k: v for k, v in remapped_state.items() if k in encoder_keys}
        missing = encoder_keys - set(load_state.keys())
        if missing:
            print(
                f"  Perhatian: {len(missing)} kunci encoder tidak dimuat dari checkpoint "
                f"(kemungkinan nama berbeda). Contoh: {list(missing)[:3]}",
                flush=True,
            )
        current_state.update(load_state)
        model.load_state_dict(current_state, strict=False)
        print(
            f"Model diinisialisasi dari {init_path} "
            f"({len(load_state)} tensor encoder dimuat, mlm_head dari acak)",
            flush=True,
        )

    if start_epoch > args.epochs:
        print("Checkpoint sudah mencapai jumlah epoch yang diminta.", flush=True)
        return

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=style_cfg.weight_decay)
    scheduler = create_linear_warmup_scheduler(optimizer, total_steps)

    print(
        f"device={device}, train_samples={len(train_loader.dataset)}, "
        f"val_samples={len(val_loader.dataset)}, total_steps={total_steps}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_metrics = run_epoch(model, val_loader, None, None, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_perplexity": train_metrics["perplexity"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_perplexity": val_metrics["perplexity"],
        })
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.2f}",
            flush=True,
        )

        # Simpan checkpoint terbaik
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                    "pretrained_from": str(init_path),
                    "history": history,
                },
                best_path,
            )
            print(f"  -> Checkpoint terbaik disimpan (val_loss={best_val_loss:.4f})", flush=True)

        # Simpan checkpoint terbaru untuk resume
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
            },
            latest_path,
        )

        # Update laporan JSON dan grafik setiap epoch
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "style_mlm",
                    "initialized_from": str(init_path),
                    "device": str(device),
                    "epochs_requested": args.epochs,
                    "epochs_completed": epoch,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_checkpoint": str(best_path),
                    "latest_checkpoint": str(latest_path),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        save_loss_plot(history, plot_path)

    print(f"\nStyle MLM terbaik disimpan di {best_path}", flush=True)
    print(f"Laporan disimpan di {report_path}", flush=True)


if __name__ == "__main__":
    main()
