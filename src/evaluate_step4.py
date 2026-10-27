"""
Step 4 - Evaluasi Model Generator Bergaya dan Classifier Clickbait.

Script ini TIDAK melatih ulang model. Ia memuat checkpoint yang sudah ada
(style_mlm.pt, classifier_pretrained.pt, classifier_random_init.pt) dan
memproduksi laporan evaluasi lengkap sesuai kriteria selesai Step 4:

    - Akurasi, macro F1, dan confusion matrix pada split test (kedua varian classifier)
    - Tabel perbandingan F1 varian pretrained vs random
    - Contoh fill-mask dari style MLM (test apakah model bisa generate secara benar)

Cara jalankan dari root repo:
    python -m src.evaluate_step4

Output:
    reports/step4_evaluation.json     - laporan evaluasi lengkap
    reports/step4_confusion_matrix.png - visualisasi confusion matrix
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import (
    CB_ID, CLS_ID, LABEL_NAMES, MASK_ID, NCB_ID, SEP_ID,
    classifier_cfg, paths, style_cfg,
)
from src.data.dataset import build_title_loader
from src.data.tokenizer import load_tokenizer
from src.model import BertForMaskedLM, BertForSequenceClassification


# ---------------------------------------------------------------------------
# Fungsi evaluasi classifier
# ---------------------------------------------------------------------------

def evaluate_classifier(model, loader, device) -> dict:
    model.eval()
    all_preds, all_truths = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_truths.extend(labels.cpu().tolist())

    y_true = np.asarray(all_truths)
    y_pred = np.asarray(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Precision dan Recall per kelas untuk detail tambahan
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        y_true, y_pred, target_names=["non-clickbait", "clickbait"], output_dict=True
    )

    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "avg_loss": float(total_loss / max(len(y_true), 1)),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["non-clickbait", "clickbait"],
        "classification_report": report_dict,
    }


def load_classifier(checkpoint_path: Path, device) -> BertForSequenceClassification:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BertForSequenceClassification().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Fungsi evaluasi style MLM – contoh fill-mask
# ---------------------------------------------------------------------------

def fill_mask_examples(model: BertForMaskedLM, tokenizer, device, n_examples: int = 6) -> list:
    """
    Buat n_examples contoh fill-mask:
    - Ambil judul dari split test, mask beberapa token acak.
    - Model diminta menebak token yang di-mask.
    - Kembalikan dict berisi judul asli, input yang di-mask, dan prediksi model.
    """
    import pandas as pd
    import random

    random.seed(42)
    test_df = pd.read_csv(paths.title_split["test"]).head(20)
    model.eval()
    examples = []

    for _, row in test_df.sample(min(n_examples, len(test_df)), random_state=42).iterrows():
        text = str(row["text"]).strip()
        label = int(row["label"])
        style_id = CB_ID if label == 1 else NCB_ID
        style_name = LABEL_NAMES[label]

        # Tokenisasi tanpa special tokens lalu tambahkan prefix gaya
        enc = tokenizer.encode(text, add_special_tokens=False)
        body = enc.ids[: style_cfg.max_len - 3]  # beri ruang [CLS] [STYLE] [SEP]
        token_ids = [CLS_ID, style_id] + body + [SEP_ID]

        original_ids = token_ids[:]

        # Mask 1-2 token acak dari bagian body (posisi 2 sampai -1)
        body_positions = list(range(2, len(token_ids) - 1))
        if not body_positions:
            continue
        n_mask = min(2, len(body_positions))
        masked_positions = sorted(random.sample(body_positions, n_mask))
        masked_ids = token_ids[:]
        for pos in masked_positions:
            masked_ids[pos] = MASK_ID

        masked_tokens_text = [tokenizer.id_to_token(original_ids[p]) for p in masked_positions]

        # Buat tensor
        max_len = style_cfg.max_len
        input_ids_t = torch.full((1, max_len), 0, dtype=torch.long, device=device)
        attn_mask_t = torch.zeros((1, max_len), dtype=torch.long, device=device)
        seq_len = len(masked_ids)
        input_ids_t[0, :seq_len] = torch.tensor(masked_ids, dtype=torch.long, device=device)
        attn_mask_t[0, :seq_len] = 1

        with torch.no_grad():
            # Forward tanpa prediction_mask → logits shape (batch, seq, vocab)
            hidden_states = model.bert(input_ids_t, attn_mask_t)
            logits = model.mlm_head(hidden_states)   # weight-tied projection

        predicted_tokens = []
        for pos in masked_positions:
            top_id = logits[0, pos].argmax().item()
            predicted_tokens.append(tokenizer.id_to_token(top_id))

        examples.append({
            "style": style_name,
            "original_text": text,
            "masked_positions": masked_positions,
            "masked_tokens": masked_tokens_text,
            "predicted_tokens": predicted_tokens,
            "correct": masked_tokens_text == predicted_tokens,
        })

    return examples


# ---------------------------------------------------------------------------
# Visualisasi confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(cm_random, cm_pretrained, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        labels = ["Non-CB", "Clickbait"]
        data_pairs = [
            (cm_random, "Random Init", "#e74c3c"),
            (cm_pretrained, "Pretrained Encoder", "#3498db"),
        ]
        for ax, (cm, title, color) in zip(axes, data_pairs):
            cm_arr = np.array(cm)
            im = ax.imshow(cm_arr, interpolation="nearest",
                           cmap=mcolors.LinearSegmentedColormap.from_list("", ["white", color]))
            ax.set_title(f"Confusion Matrix\n{title}", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            for i in range(2):
                for j in range(2):
                    val = cm_arr[i, j]
                    ax.text(j, i, str(val), ha="center", va="center",
                            color="white" if val > cm_arr.max() / 2 else "black",
                            fontsize=13, fontweight="bold")

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix disimpan di {output_path}", flush=True)
    except ImportError:
        print("matplotlib tidak tersedia, plot confusion matrix dilewati.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Evaluasi classifier clickbait dan style MLM dari checkpoint"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=classifier_cfg.batch_size)
    parser.add_argument(
        "--classifier-random",
        default=str(paths.checkpoints_dir / "classifier_random_init.pt"),
    )
    parser.add_argument(
        "--classifier-pretrained",
        default=str(paths.checkpoints_dir / "classifier_pretrained.pt"),
    )
    parser.add_argument(
        "--style-mlm",
        default=str(paths.checkpoints_dir / "style_mlm.pt"),
    )
    parser.add_argument(
        "--fill-mask-examples", type=int, default=6,
        help="Jumlah contoh fill-mask untuk ditampilkan di laporan"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia.")

    print(f"Evaluasi Step 4 — device={device}", flush=True)

    # ------------------------------------------------------------------ #
    #  1. Evaluasi classifier pada split test                             #
    # ------------------------------------------------------------------ #
    test_loader = build_title_loader(
        "test", batch_size=args.batch_size, with_style=False, shuffle=False
    )
    print(f"\nTest split: {len(test_loader.dataset)} sampel", flush=True)

    print("\n[1/3] Memuat classifier_random_init ...", flush=True)
    random_model = load_classifier(Path(args.classifier_random), device)
    random_metrics = evaluate_classifier(random_model, test_loader, device)

    print("[2/3] Memuat classifier_pretrained ...", flush=True)
    pretrained_model = load_classifier(Path(args.classifier_pretrained), device)
    pretrained_metrics = evaluate_classifier(pretrained_model, test_loader, device)

    # ------------------------------------------------------------------ #
    #  2. Fill-mask dari style MLM                                        #
    # ------------------------------------------------------------------ #
    fill_mask_results = []
    style_mlm_path = Path(args.style_mlm)
    if style_mlm_path.exists():
        print("[3/3] Memuat style_mlm untuk contoh fill-mask ...", flush=True)
        ckpt = torch.load(style_mlm_path, map_location=device)
        style_model = BertForMaskedLM().to(device)
        style_model.load_state_dict(ckpt["model_state"])
        tokenizer = load_tokenizer()
        fill_mask_results = fill_mask_examples(style_model, tokenizer, device, args.fill_mask_examples)
    else:
        print(f"[3/3] style_mlm.pt tidak ditemukan di {style_mlm_path}, melewati fill-mask.", flush=True)

    # ------------------------------------------------------------------ #
    #  3. Buat laporan                                                    #
    # ------------------------------------------------------------------ #
    comparison_table = [
        {
            "variant": "random_init",
            "test_accuracy": random_metrics["accuracy"],
            "test_f1_macro": random_metrics["f1_macro"],
            "confusion_matrix": random_metrics["confusion_matrix"],
        },
        {
            "variant": "pretrained",
            "test_accuracy": pretrained_metrics["accuracy"],
            "test_f1_macro": pretrained_metrics["f1_macro"],
            "confusion_matrix": pretrained_metrics["confusion_matrix"],
        },
    ]

    delta_f1 = pretrained_metrics["f1_macro"] - random_metrics["f1_macro"]
    report = {
        "step": 4,
        "device": str(device),
        "test_split_samples": len(test_loader.dataset),
        "classifier_random_init": {
            "checkpoint": args.classifier_random,
            "test_metrics": random_metrics,
        },
        "classifier_pretrained": {
            "checkpoint": args.classifier_pretrained,
            "test_metrics": pretrained_metrics,
        },
        "comparison_table": comparison_table,
        "delta_f1_pretrained_minus_random": round(delta_f1, 6),
        "interpretation": (
            "Pretraining memberikan manfaat (F1 lebih tinggi)" if delta_f1 > 0
            else "Pretraining tidak meningkatkan F1 pada konfigurasi ini (epoch mungkin terlalu sedikit)"
        ),
        "fill_mask_examples": fill_mask_results,
    }

    report_path = paths.reports_dir / "step4_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    save_confusion_matrix_plot(
        random_metrics["confusion_matrix"],
        pretrained_metrics["confusion_matrix"],
        paths.reports_dir / "step4_confusion_matrix.png",
    )

    # ------------------------------------------------------------------ #
    #  4. Tampilkan ringkasan di terminal                                 #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55, flush=True)
    print("  STEP 4 - HASIL EVALUASI TEST SPLIT", flush=True)
    print("=" * 55, flush=True)
    print(f"{'Variant':<20} {'Accuracy':>10} {'Macro F1':>10}", flush=True)
    print("-" * 44, flush=True)
    for row in comparison_table:
        print(
            f"{row['variant']:<20} {row['test_accuracy']:>10.4f} {row['test_f1_macro']:>10.4f}",
            flush=True,
        )
    print("-" * 44, flush=True)
    print(
        f"Delta F1 (pretrained - random): {delta_f1:+.4f}  =>  {report['interpretation']}",
        flush=True,
    )
    print("=" * 55, flush=True)
    print(f"\nLaporan lengkap: {report_path}", flush=True)

    if fill_mask_results:
        print("\n--- Contoh Fill-Mask dari Style MLM ---", flush=True)
        for i, ex in enumerate(fill_mask_results, 1):
            correct_mark = "OK" if ex["correct"] else "X"
            print(
                f"  [{i}] [{ex['style'].upper()[:2]}] {ex['original_text'][:60]}", flush=True
            )
            for masked, predicted in zip(ex["masked_tokens"], ex["predicted_tokens"]):
                match = "[OK]" if masked == predicted else "[X]"
                print(f"       masked='{masked}' -> pred='{predicted}' {match}", flush=True)


if __name__ == "__main__":
    main()
