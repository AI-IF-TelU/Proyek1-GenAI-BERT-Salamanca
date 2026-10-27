import argparse
import json
import math
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import SEED, paths, pretrain_cfg
from src.collator import DynamicMaskingCollator
from src.data.dataset import build_pretrain_loader
from src.model import BertForMaskedLM


def save_loss_plot(history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train"]["loss"] for item in history]
    val_losses = [item["validation"]["loss"] for item in history]
    figure, axis = plt.subplots()
    axis.plot(epochs, train_losses, marker="o", label="train")
    axis.plot(epochs, val_losses, marker="o", label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked-token cross-entropy loss")
    axis.set_title("IndoMiniBERT MLM pretraining loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_scheduler(optimizer, total_steps):
    warmup_steps = int(total_steps * pretrain_cfg.warmup_ratio)

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        return max(0.0, float(total_steps - step) / max(total_steps - warmup_steps, 1))

    return LambdaLR(optimizer, multiplier)


def run_epoch(model, loader, optimizer, scheduler, device, scaler=None, max_steps=None):
    training = optimizer is not None
    model.train(training)
    criterion = CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    steps = 0

    progress = tqdm(loader, desc="train" if training else "validation", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        prediction_mask = labels.ne(-100)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda" and scaler is not None and scaler.is_enabled(),
            ):
                logits = model(input_ids, attention_mask, prediction_mask=prediction_mask)
                targets = labels[prediction_mask]
                loss = criterion(logits, targets)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
                    optimizer.step()
                scheduler.step()

        token_count = int(prediction_mask.sum().item())
        total_loss += loss.item() * token_count
        total_tokens += token_count
        correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": correct / max(total_tokens, 1),
        "masked_tokens": total_tokens,
        "steps": steps,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain IndoMiniBERT with masked language modeling")
    parser.add_argument("--epochs", type=int, default=pretrain_cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=pretrain_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=pretrain_cfg.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="resume from the latest epoch checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="limit train steps per epoch for smoke tests")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia pada environment ini")
    set_seed()

    train_loader = build_pretrain_loader(
        "train",
        batch_size=args.batch_size,
        num_workers=pretrain_cfg.num_workers,
        collate_fn=DynamicMaskingCollator(),
    )
    val_loader = build_pretrain_loader(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=pretrain_cfg.num_workers,
        collate_fn=DynamicMaskingCollator(),
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    model = BertForMaskedLM().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=pretrain_cfg.weight_decay)
    scheduler = create_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and pretrain_cfg.use_amp
    )
    latest_path = paths.checkpoints_dir / "pretrain_mlm_latest.pt"
    best_path = paths.checkpoints_dir / "pretrain_mlm.pt"
    start_epoch = 1
    best_val_loss = math.inf
    history = []

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Checkpoint resume tidak ditemukan: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        print(f"Resume dari epoch {start_epoch}", flush=True)

    if start_epoch > args.epochs:
        print("Checkpoint sudah mencapai jumlah epoch yang diminta.", flush=True)
        return

    print(
        f"device={device}, train_blocks={len(train_loader.dataset)}, "
        f"val_blocks={len(val_loader.dataset)}, total_steps={total_steps}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, args.max_steps
        )
        val_metrics = run_epoch(model, val_loader, None, None, device, max_steps=args.max_steps)
        result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(result)
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "validation": val_metrics},
                best_path,
            )
        save_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, epoch, best_val_loss, history
        )

        report_path = paths.reports_dir / "pretrain_mlm_metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "device": str(device),
                    "epochs_requested": args.epochs,
                    "epochs_completed": epoch,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "training_blocks": len(train_loader.dataset),
                    "validation_blocks": len(val_loader.dataset),
                    "history": history,
                    "best_checkpoint": str(best_path),
                    "latest_checkpoint": str(latest_path),
                },
                report_file,
                indent=2,
                ensure_ascii=False,
            )
        save_loss_plot(history, paths.reports_dir / "pretrain_mlm_loss.png")

    print(f"Checkpoint terbaik disimpan di {best_path}", flush=True)


if __name__ == "__main__":
    main()
