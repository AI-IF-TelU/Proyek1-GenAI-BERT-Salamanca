"""
Step 4 - Training Classifier Clickbait (Ablation Pretrained vs Random Init).

Melatih dua varian classifier:
  1. random_init  - BertForSequenceClassification dari inisialisasi acak
  2. pretrained   - encoder diisi dari checkpoint pretraining Step 3, classification
                    head tetap acak (ablation study)

Cara jalankan dari root repo:
    python -m src.classifier
    python -m src.classifier --epochs 5 --batch-size 64 --device cuda

Output:
    checkpoints/classifier_random_init.pt       - bobot classifier acak terbaik
    checkpoints/classifier_pretrained.pt        - bobot classifier pretrained terbaik
    reports/style_classifier_ablation.json      - akurasi, macro F1, confusion matrix
    reports/classifier_loss.png                 - grafik train/val loss kedua varian
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import SEED, classifier_cfg, paths, pretrain_cfg
from src.data.dataset import build_title_loader
from src.model import BertForSequenceClassification


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_linear_warmup_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.06):
    warmup_steps = int(total_steps * warmup_ratio)

    def _lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return max(0.0, float(total_steps - step) / max(total_steps - warmup_steps, 1))

    return LambdaLR(optimizer, _lr_lambda)


def evaluate(model, loader, device):
    """Evaluasi model: loss, accuracy, macro F1, confusion matrix."""
    model.eval()
    predictions, truths, losses = [], [], []
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            losses.append(loss.item() * labels.size(0))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(labels.cpu().tolist())

    y_true = np.asarray(truths)
    y_pred = np.asarray(predictions)
    return {
        "loss": float(np.sum(losses) / max(len(loader.dataset), 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": len(y_true),
    }


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Satu epoch training, kembalikan rata-rata loss."""
    model.train()
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(input_ids, attention_mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(len(loader.dataset), 1)


def train_classifier(model, train_loader, val_loader, device, epochs, lr):
    """Training loop lengkap; simpan state terbaik berdasarkan val F1."""
    total_steps = max(1, len(train_loader) * epochs)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=classifier_cfg.weight_decay)
    scheduler = create_linear_warmup_scheduler(optimizer, total_steps)

    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        })
        print(
            f"  epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1_macro']:.4f}",
            flush=True,
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint classifier")
    model.load_state_dict(best_state)
    return history, best_val_f1


def load_pretrained_encoder(model, checkpoint_path: Path, device):
    """
    Muat bobot encoder BERT dari checkpoint pretraining (bukan bobot classifier).

    Mendukung dua format checkpoint:
      - Format lama (Step 3): kunci attention.query/key/value/output, intermediate, output_norm
      - Format baru (Step 4+): kunci attention.q_proj/k_proj/v_proj/out_proj, ffn.net, ffn_norm
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Format checkpoint tidak valid: {checkpoint_path}")

    # Remap dari format lama ke format baru jika perlu
    remap = {
        "attention.query.": "attention.q_proj.",
        "attention.key.":   "attention.k_proj.",
        "attention.value.": "attention.v_proj.",
        "attention.output.": "attention.out_proj.",
        "intermediate.":    "ffn.net.0.",
    }
    remapped = {}
    for key, val in model_state.items():
        new_key = key
        for old_sfx, new_sfx in remap.items():
            if old_sfx in new_key:
                new_key = new_key.replace(old_sfx, new_sfx)
                break
        if ".output.weight" in new_key and "attention" not in new_key:
            new_key = new_key.replace(".output.weight", ".ffn.net.3.weight")
        if ".output.bias" in new_key and "attention" not in new_key:
            new_key = new_key.replace(".output.bias", ".ffn.net.3.bias")
        if ".output_norm." in new_key:
            new_key = new_key.replace(".output_norm.", ".ffn_norm.")
        remapped[new_key] = val

    # Hanya ambil kunci encoder (bert.*)
    encoder_state = {
        key.removeprefix("bert."): value
        for key, value in remapped.items()
        if key.startswith("bert.")
    }
    if not encoder_state:
        raise ValueError(
            f"Checkpoint tidak berisi bobot encoder BERT (prefix 'bert.'): {checkpoint_path}"
        )
    model.bert.load_state_dict(encoder_state, strict=False)
    print(f"  -> Encoder dimuat dari {checkpoint_path} ({len(encoder_state)} tensor)", flush=True)


def save_model(model, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)
    print(f"  -> Checkpoint disimpan di {path}", flush=True)


def save_loss_plot(results: dict, output_path: Path) -> None:
    """Simpan grafik train/val loss dua varian sebagai PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"random_init": ("#e74c3c", "#c0392b"), "pretrained": ("#3498db", "#2980b9")}
        for name, data in results.items():
            history = data["history"]
            epochs = [h["epoch"] for h in history]
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]
            c_train, c_val = colors.get(name, ("#555", "#333"))
            label = "Random Init" if name == "random_init" else "Pretrained Encoder"
            ax.plot(epochs, train_losses, color=c_train, linestyle="--", marker="o",
                    markersize=4, label=f"{label} – train")
            ax.plot(epochs, val_losses, color=c_val, linestyle="-", marker="s",
                    markersize=4, label=f"{label} – validasi")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Classification Cross-Entropy Loss")
        ax.set_title("Clickbait Classifier – Ablation: Pretrained vs Random Init")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Grafik loss disimpan di {output_path}", flush=True)
    except ImportError:
        print("matplotlib tidak tersedia, grafik tidak disimpan.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Latih classifier clickbait dalam dua varian "
            "(pretrained vs random-init) untuk ablation study"
        )
    )
    parser.add_argument("--epochs", type=int, default=classifier_cfg.epochs,
                        help=f"Jumlah epoch (default: {classifier_cfg.epochs})")
    parser.add_argument("--batch-size", type=int, default=classifier_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=classifier_cfg.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pretrained-checkpoint",
        default=str(paths.checkpoints_dir / "pretrain_mlm.pt"),
        help="Checkpoint Step 3 untuk mengisi encoder model pretrained",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia pada environment ini")
    pretrained_path = Path(args.pretrained_checkpoint)
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Checkpoint Step 3 tidak ditemukan di: {pretrained_path}\n"
            "Penyebab: File checkpoint tidak ikut terbawa saat git clone (ada di .gitignore).\n"
            "Solusi:\n"
            "  1. Upload file 'checkpoints/pretrain_mlm.pt' (~29.5 MB) dari laptop ke folder checkpoints/ di Colab.\n"
            "  2. Atau jalankan pretraining terlebih dahulu: python -m src.pretrain --epochs 5 --device cuda\n"
        )

    set_seed()

    # Buat dua DataLoader terpisah dengan seed yang sama agar data urutan sama
    train_loaders = [
        build_title_loader(
            "train",
            batch_size=args.batch_size,
            with_style=False,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        for _ in range(2)
    ]
    val_loader = build_title_loader("val", batch_size=args.batch_size, with_style=False, shuffle=False)
    test_loader = build_title_loader("test", batch_size=args.batch_size, with_style=False, shuffle=False)

    print(
        f"device={device}  "
        f"train={len(train_loaders[0].dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)} samples",
        flush=True,
    )

    # --- Varian 1: Random Init ---
    print("\n[1/2] Training varian random_init...", flush=True)
    random_model = BertForSequenceClassification().to(device)
    random_history, random_best_val_f1 = train_classifier(
        random_model, train_loaders[0], val_loader, device, args.epochs, args.lr
    )
    random_test_metrics = evaluate(random_model, test_loader, device)
    random_checkpoint = paths.checkpoints_dir / "classifier_random_init.pt"
    save_model(
        random_model,
        random_checkpoint,
        {
            "variant": "random_init",
            "pretrained_from": None,
            "best_val_f1_macro": random_best_val_f1,
            "test_metrics": random_test_metrics,
        },
    )

    # --- Varian 2: Pretrained Encoder ---
    print("\n[2/2] Training varian pretrained...", flush=True)
    pretrained_model = BertForSequenceClassification().to(device)
    load_pretrained_encoder(pretrained_model, pretrained_path, device)
    pretrained_history, pretrained_best_val_f1 = train_classifier(
        pretrained_model, train_loaders[1], val_loader, device, args.epochs, args.lr
    )
    pretrained_test_metrics = evaluate(pretrained_model, test_loader, device)
    pretrained_checkpoint = paths.checkpoints_dir / "classifier_pretrained.pt"
    save_model(
        pretrained_model,
        pretrained_checkpoint,
        {
            "variant": "pretrained",
            "pretrained_from": str(pretrained_path),
            "best_val_f1_macro": pretrained_best_val_f1,
            "test_metrics": pretrained_test_metrics,
        },
    )

    # --- Laporan Ablation ---
    results = {
        "random_init": {
            "history": random_history,
            "best_val_f1_macro": random_best_val_f1,
            "test_metrics": random_test_metrics,
            "checkpoint": str(random_checkpoint),
        },
        "pretrained": {
            "history": pretrained_history,
            "best_val_f1_macro": pretrained_best_val_f1,
            "test_metrics": pretrained_test_metrics,
            "checkpoint": str(pretrained_checkpoint),
        },
    }

    report = {
        "model": "clickbait_classifier",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pretrained_checkpoint": str(pretrained_path),
        "test_split_samples": len(test_loader.dataset),
        "random_init": results["random_init"],
        "pretrained": results["pretrained"],
        # Tabel perbandingan ringkas sesuai kriteria selesai Step 4
        "comparison_table": [
            {
                "variant": name,
                "test_accuracy": data["test_metrics"]["accuracy"],
                "test_f1_macro": data["test_metrics"]["f1_macro"],
                "test_confusion_matrix": data["test_metrics"]["confusion_matrix"],
            }
            for name, data in results.items()
        ],
    }

    report_path = paths.reports_dir / "style_classifier_ablation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    save_loss_plot(results, paths.reports_dir / "classifier_loss.png")

    print(f"\nLaporan ablation disimpan di {report_path}", flush=True)
    print("\n=== TABEL PERBANDINGAN ABLATION ===", flush=True)
    print(f"{'Variant':<18} {'Test Acc':>10} {'Test F1':>10}", flush=True)
    print("-" * 42, flush=True)
    for row in report["comparison_table"]:
        print(
            f"{row['variant']:<18} {row['test_accuracy']:>10.4f} {row['test_f1_macro']:>10.4f}",
            flush=True,
        )
    print("", flush=True)
    delta = pretrained_test_metrics["f1_macro"] - random_test_metrics["f1_macro"]
    print(
        f"Delta F1 (pretrained - random): {delta:+.4f} "
        f"({'pretraining bermanfaat' if delta > 0 else 'pretraining tidak bermanfaat di sini'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
