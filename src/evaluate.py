"""
Step 5 - Evaluasi Hasil Generasi Judul IndoMiniBERT.

Script ini mengevaluasi kualitas judul yang dihasilkan oleh src/generate.py
dengan empat metrik utama:

  1. NOVELTY (Kebaruan)
     Persentase judul yang tidak muncul persis di data training.
     Tujuan: memastikan model tidak hanya menghafal data training.

  2. DIVERSITY (Keberagaman)
     a. Distinct-1 & Distinct-2: rasio unigram/bigram unik dari total token.
     b. Self-BLEU (rata-rata BLEU setiap judul terhadap judul lain):
        semakin rendah -> semakin beragam.

  3. CLASSIFIER-BASED QUALITY (Kualitas Gaya)
     Setiap judul yang di-generate di-prediksi oleh classifier clickbait.
     Style accuracy = persentase judul yang berhasil diprediksi sesuai gaya targetnya.
     Menggunakan checkpoints/classifier_pretrained.pt dari Step 4.

  4. PERPLEXITY (Kelancaran)
     Rata-rata perplexity judul terhadap style_mlm.pt dengan dynamic masking
     ringan. Perplexity lebih rendah = model lebih "yakin" dengan judul tersebut.

Cara jalankan (dari root repo):
    python -m src.evaluate
    python -m src.evaluate --input reports/generated_titles.json
    python -m src.evaluate --no-classifier   # lewati evaluasi classifier

Output:
    reports/step5_evaluation.json    - laporan evaluasi lengkap JSON
    reports/step5_evaluation.txt     - ringkasan teks mudah dibaca

Catatan:
    Script ini membutuhkan file reports/generated_titles.json dari src/generate.py.
    Jika file belum ada, jalankan src.generate terlebih dahulu.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch

from config import (
    CB_ID, CLS_ID, MASK_ID, NCB_ID, PAD_ID, SEP_ID,
    LABEL_NAMES, SEED,
    classifier_cfg, paths, style_cfg,
)
from src.data.tokenizer import load_tokenizer
from src.model import BertForMaskedLM, BertForSequenceClassification


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_text(tokenizer, text: str) -> List[int]:
    """Tokenisasi teks dan kembalikan daftar token ID."""
    enc = tokenizer.encode(text, add_special_tokens=False)
    return enc.ids


def load_training_titles(splits_dir: Path) -> set:
    """Muat semua judul dari split training sebagai set untuk cek novelty."""
    import pandas as pd
    train_csv = splits_dir / "titles_train.csv"
    if not train_csv.exists():
        print(f"  [WARNING] File training tidak ditemukan: {train_csv}")
        return set()
    df = pd.read_csv(train_csv)
    texts = set()
    for t in df["text"].dropna():
        texts.add(str(t).strip().lower())
    return texts


# ---------------------------------------------------------------------------
# Metrik 1: Novelty
# ---------------------------------------------------------------------------

def compute_novelty(generated_texts: List[str], train_texts: set) -> dict:
    """
    Hitung persentase judul yang tidak ada di training set.

    Returns:
        dict dengan novel_count, total, novelty_ratio
    """
    novel_count = 0
    details = []
    for text in generated_texts:
        is_novel = text.strip().lower() not in train_texts
        if is_novel:
            novel_count += 1
        details.append({"text": text, "is_novel": is_novel})

    total = len(generated_texts)
    return {
        "novel_count": novel_count,
        "total": total,
        "novelty_ratio": round(novel_count / max(total, 1), 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Metrik 2: Diversity (Distinct-1, Distinct-2, Self-BLEU)
# ---------------------------------------------------------------------------

def compute_distinct(token_lists: List[List[int]]) -> dict:
    """
    Distinct-1 dan Distinct-2 mengukur keberagaman leksikal.
    Distinct-N = (jumlah N-gram unik) / (total N-gram)
    Nilai lebih tinggi = lebih beragam.
    """
    unigrams_all = []
    bigrams_all = []
    for ids in token_lists:
        unigrams_all.extend(ids)
        bigrams_all.extend(zip(ids[:-1], ids[1:]))

    d1 = len(set(unigrams_all)) / max(len(unigrams_all), 1)
    d2 = len(set(bigrams_all)) / max(len(bigrams_all), 1)

    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "total_unigrams": len(unigrams_all),
        "unique_unigrams": len(set(unigrams_all)),
        "total_bigrams": len(bigrams_all),
        "unique_bigrams": len(set(bigrams_all)),
    }


def sentence_bleu_score(hypothesis: List[int], references: List[List[int]], max_n: int = 4) -> float:
    """
    Hitung BLEU-score sederhana antara satu hipotesis dan beberapa referensi.
    Implementasi sederhana tanpa paket nltk.
    """
    if not hypothesis or not references:
        return 0.0

    scores = []
    for n in range(1, min(max_n, len(hypothesis)) + 1):
        hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1))
        max_ref_counts: Counter = Counter()
        for ref in references:
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            for gram, count in ref_ngrams.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], count)

        clip_count = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_ngrams.items())
        total_count = sum(hyp_ngrams.values())
        precision = clip_count / max(total_count, 1)
        scores.append(precision)

    if not scores or all(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(max(s, 1e-9)) for s in scores) / len(scores)
    return math.exp(log_avg)


def compute_self_bleu(token_lists: List[List[int]], sample_size: int = 50) -> dict:
    """
    Hitung Self-BLEU: rata-rata BLEU setiap kalimat terhadap semua kalimat lain.
    Nilai lebih rendah = lebih beragam.
    Untuk efisiensi, subsampel jika jumlah terlalu besar.
    """
    if len(token_lists) < 2:
        return {"self_bleu": None, "note": "kurang dari 2 judul untuk dihitung"}

    indices = list(range(len(token_lists)))
    if len(indices) > sample_size:
        random.shuffle(indices)
        indices = indices[:sample_size]

    bleu_scores = []
    for i in indices:
        hypothesis = token_lists[i]
        references = [token_lists[j] for j in range(len(token_lists)) if j != i]
        bleu_scores.append(sentence_bleu_score(hypothesis, references))

    return {
        "self_bleu": round(float(np.mean(bleu_scores)), 4),
        "self_bleu_std": round(float(np.std(bleu_scores)), 4),
        "n_evaluated": len(bleu_scores),
    }


def compute_diversity(generated_texts: List[str], tokenizer) -> dict:
    """Gabungkan semua metrik diversity."""
    token_lists = [tokenize_text(tokenizer, t) for t in generated_texts]
    token_lists = [t for t in token_lists if t]  # buang kosong

    distinct = compute_distinct(token_lists)
    self_bleu = compute_self_bleu(token_lists)

    return {
        "n_titles": len(generated_texts),
        **distinct,
        **self_bleu,
    }


# ---------------------------------------------------------------------------
# Metrik 3: Classifier-Based Style Accuracy
# ---------------------------------------------------------------------------

def evaluate_style_accuracy(
    generated: List[dict],
    tokenizer,
    device: torch.device,
    classifier_path: Optional[Path] = None,
) -> dict:
    """
    Evaluasi apakah judul yang dihasilkan berhasil mewakili gaya target.
    Menggunakan classifier clickbait dari Step 4.

    Returns:
        dict dengan style_accuracy, per_style breakdown, predictions
    """
    if classifier_path is None or not classifier_path.exists():
        return {
            "style_accuracy": None,
            "note": f"Classifier tidak ditemukan di {classifier_path}",
        }

    # Muat classifier
    ckpt = torch.load(classifier_path, map_location=device)
    classifier = BertForSequenceClassification().to(device)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()

    predictions = []
    correct = 0
    cb_correct, cb_total = 0, 0
    ncb_correct, ncb_total = 0, 0

    max_len = style_cfg.max_len

    with torch.no_grad():
        for item in generated:
            text = item.get("generated_text", "")
            target_style = item.get("style", "")
            target_label = 1 if target_style == "clickbait" else 0

            # Tokenisasi: [CLS] judul [SEP] (tanpa token gaya)
            enc = tokenizer.encode(text, add_special_tokens=False)
            body_ids = enc.ids[: max_len - 2]
            seq = [CLS_ID] + body_ids + [SEP_ID]
            seq_len = len(seq)

            input_ids = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
            attn_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)
            input_ids[0, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[0, :seq_len] = 1

            logits = classifier(input_ids, attn_mask)
            pred_label = int(logits.argmax(dim=-1).item())
            pred_style = LABEL_NAMES[pred_label]

            is_correct = pred_label == target_label
            if is_correct:
                correct += 1

            if target_label == 1:
                cb_total += 1
                if is_correct:
                    cb_correct += 1
            else:
                ncb_total += 1
                if is_correct:
                    ncb_correct += 1

            predictions.append({
                "text": text,
                "target_style": target_style,
                "predicted_style": pred_style,
                "correct": is_correct,
            })

    total = len(generated)
    return {
        "style_accuracy": round(correct / max(total, 1), 4),
        "clickbait_accuracy": round(cb_correct / max(cb_total, 1), 4) if cb_total else None,
        "non_clickbait_accuracy": round(ncb_correct / max(ncb_total, 1), 4) if ncb_total else None,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Metrik 4: Perplexity (Kelancaran Model)
# ---------------------------------------------------------------------------

def compute_generation_perplexity(
    generated: List[dict],
    tokenizer,
    device: torch.device,
    style_mlm_path: Optional[Path] = None,
) -> dict:
    """
    Hitung rata-rata perplexity judul terhadap style_mlm.pt.
    Menggunakan masking 15% dan menghitung loss rata-rata sebagai proxy perplexity.
    Nilai lebih rendah = model lebih yakin = lebih "lancar".

    Returns:
        dict dengan mean_perplexity, per_title list
    """
    if style_mlm_path is None or not style_mlm_path.exists():
        return {
            "mean_perplexity": None,
            "note": f"style_mlm.pt tidak ditemukan di {style_mlm_path}",
        }

    # Muat model
    ckpt = torch.load(style_mlm_path, map_location=device)
    model = BertForMaskedLM().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    perplexities = []
    per_title = []

    mask_prob = 0.15
    max_len = style_cfg.max_len

    with torch.no_grad():
        for item in generated:
            text = item.get("generated_text", "")
            target_style = item.get("style", "")
            style_id = CB_ID if target_style == "clickbait" else NCB_ID

            enc = tokenizer.encode(text, add_special_tokens=False)
            body_ids = enc.ids[: max_len - 3]
            seq = [CLS_ID, style_id] + body_ids + [SEP_ID]
            seq_len = len(seq)

            if seq_len <= 3:
                continue  # judul kosong, lewati

            input_ids_orig = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            input_ids_masked = input_ids_orig.clone()
            labels = torch.full_like(input_ids_orig, -100)

            # Mask 15% body token secara acak
            body_positions = list(range(2, seq_len - 1))
            mask_count = max(1, int(mask_prob * len(body_positions)))
            masked_pos = random.sample(body_positions, min(mask_count, len(body_positions)))
            for pos in masked_pos:
                labels[0, pos] = input_ids_orig[0, pos]
                input_ids_masked[0, pos] = MASK_ID

            # Buat attention mask
            attn_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)
            attn_mask[0, :seq_len] = 1

            # Padding
            input_padded = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
            input_padded[0, :seq_len] = input_ids_masked[0]

            labels_padded = torch.full((1, max_len), -100, dtype=torch.long, device=device)
            labels_padded[0, :seq_len] = labels[0]

            # Forward
            prediction_mask = labels_padded.ne(-100)
            hidden = model.bert(input_padded, attn_mask)
            logits = model.mlm_head(hidden)

            if prediction_mask.any():
                loss = criterion(
                    logits[0][prediction_mask[0]],
                    labels_padded[0][prediction_mask[0]],
                )
                ppl = math.exp(min(loss.item(), 100))
                perplexities.append(ppl)
                per_title.append({"text": text, "perplexity": round(ppl, 2)})

    if not perplexities:
        return {"mean_perplexity": None, "note": "Tidak ada judul yang bisa dievaluasi"}

    return {
        "mean_perplexity": round(float(np.mean(perplexities)), 4),
        "median_perplexity": round(float(np.median(perplexities)), 4),
        "min_perplexity": round(float(np.min(perplexities)), 4),
        "max_perplexity": round(float(np.max(perplexities)), 4),
        "n_evaluated": len(perplexities),
        "per_title": per_title,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Evaluasi hasil generasi judul IndoMiniBERT"
    )
    parser.add_argument(
        "--input",
        default=str(paths.reports_dir / "generated_titles.json"),
        help="File JSON hasil generasi dari src.generate (default: reports/generated_titles.json)",
    )
    parser.add_argument(
        "--style-mlm",
        default=str(paths.checkpoints_dir / "style_mlm.pt"),
        help="Checkpoint style_mlm.pt untuk evaluasi perplexity",
    )
    parser.add_argument(
        "--classifier",
        default=str(paths.checkpoints_dir / "classifier_pretrained.pt"),
        help="Checkpoint classifier untuk evaluasi style accuracy",
    )
    parser.add_argument(
        "--no-classifier", action="store_true",
        help="Lewati evaluasi classifier (lebih cepat)",
    )
    parser.add_argument(
        "--no-perplexity", action="store_true",
        help="Lewati evaluasi perplexity (lebih cepat)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-json",
        default=str(paths.reports_dir / "step5_evaluation.json"),
    )
    parser.add_argument(
        "--output-txt",
        default=str(paths.reports_dir / "step5_evaluation.txt"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Muat hasil generasi
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"File hasil generasi tidak ditemukan: {input_path}\n"
            "Jalankan terlebih dahulu: python -m src.generate"
        )

    with open(input_path, encoding="utf-8") as f:
        gen_report = json.load(f)

    generated: List[dict] = gen_report.get("generated", [])
    if not generated:
        raise ValueError("File generasi kosong, tidak ada judul untuk dievaluasi.")

    print(f"Evaluasi Step 5 — {len(generated)} judul dari {input_path}", flush=True)
    print(f"Device: {device}", flush=True)

    # Pisahkan teks dari generated
    all_texts = [r["generated_text"] for r in generated if r.get("generated_text")]
    cb_texts = [r["generated_text"] for r in generated if r.get("style") == "clickbait" and r.get("generated_text")]
    ncb_texts = [r["generated_text"] for r in generated if r.get("style") == "non-clickbait" and r.get("generated_text")]

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------ #
    # 1. Novelty                                                          #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Menghitung Novelty...", flush=True)
    train_titles = load_training_titles(paths.splits_dir)
    novelty_all = compute_novelty(all_texts, train_titles)
    novelty_cb = compute_novelty(cb_texts, train_titles)
    novelty_ncb = compute_novelty(ncb_texts, train_titles)
    print(
        f"  Novelty: {novelty_all['novelty_ratio']:.2%} "
        f"({novelty_all['novel_count']}/{novelty_all['total']} judul baru)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Diversity                                                        #
    # ------------------------------------------------------------------ #
    print("[2/4] Menghitung Diversity...", flush=True)
    diversity_all = compute_diversity(all_texts, tokenizer)
    diversity_cb = compute_diversity(cb_texts, tokenizer) if cb_texts else {}
    diversity_ncb = compute_diversity(ncb_texts, tokenizer) if ncb_texts else {}
    print(
        f"  Distinct-1: {diversity_all['distinct_1']:.4f}  "
        f"Distinct-2: {diversity_all['distinct_2']:.4f}  "
        f"Self-BLEU: {diversity_all.get('self_bleu', 'N/A')}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 3. Style Accuracy (Classifier)                                      #
    # ------------------------------------------------------------------ #
    print("[3/4] Menghitung Style Accuracy...", flush=True)
    if args.no_classifier:
        style_results = {"style_accuracy": None, "note": "Dilewati (--no-classifier)"}
        print("  Dilewati.", flush=True)
    else:
        style_results = evaluate_style_accuracy(
            generated, tokenizer, device, Path(args.classifier)
        )
        acc = style_results.get("style_accuracy")
        if acc is not None:
            print(f"  Style Accuracy: {acc:.2%}", flush=True)
        else:
            print(f"  {style_results.get('note', '')}", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Perplexity                                                       #
    # ------------------------------------------------------------------ #
    print("[4/4] Menghitung Perplexity...", flush=True)
    if args.no_perplexity:
        ppl_results = {"mean_perplexity": None, "note": "Dilewati (--no-perplexity)"}
        print("  Dilewati.", flush=True)
    else:
        ppl_results = compute_generation_perplexity(
            generated, tokenizer, device, Path(args.style_mlm)
        )
        mean_ppl = ppl_results.get("mean_perplexity")
        if mean_ppl is not None:
            print(f"  Mean Perplexity: {mean_ppl:.2f}", flush=True)
        else:
            print(f"  {ppl_results.get('note', '')}", flush=True)

    # ------------------------------------------------------------------ #
    # Buat laporan                                                        #
    # ------------------------------------------------------------------ #
    report = {
        "step": 5,
        "device": str(device),
        "input_file": str(input_path),
        "n_generated": len(generated),
        "generation_config": {
            "model": gen_report.get("model"),
            "checkpoint": gen_report.get("checkpoint"),
            "iterations": gen_report.get("iterations"),
            "top_k": gen_report.get("top_k"),
            "temperature": gen_report.get("temperature"),
        },
        "metrics": {
            "novelty": {
                "all": {k: v for k, v in novelty_all.items() if k != "details"},
                "clickbait": {k: v for k, v in novelty_cb.items() if k != "details"},
                "non_clickbait": {k: v for k, v in novelty_ncb.items() if k != "details"},
            },
            "diversity": {
                "all": diversity_all,
                "clickbait": diversity_cb,
                "non_clickbait": diversity_ncb,
            },
            "style_accuracy": {k: v for k, v in style_results.items() if k != "predictions"},
            "perplexity": {k: v for k, v in ppl_results.items() if k != "per_title"},
        },
        "generated_titles": [
            {"style": r.get("style"), "text": r.get("generated_text"), "confidence": r.get("avg_confidence")}
            for r in generated
        ],
    }

    # Simpan JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nLaporan JSON disimpan di: {output_json}", flush=True)

    # Simpan TXT ringkasan
    output_txt = Path(args.output_txt)
    lines = [
        "=" * 60,
        "  STEP 5 - EVALUASI HASIL GENERASI IndoMiniBERT",
        "=" * 60,
        f"Jumlah judul yang dievaluasi: {len(generated)}",
        "",
        "--- 1. NOVELTY (Kebaruan) ---",
        f"  Semua gaya   : {novelty_all['novelty_ratio']:.2%} ({novelty_all['novel_count']}/{novelty_all['total']})",
        f"  Clickbait    : {novelty_cb['novelty_ratio']:.2%}  ({novelty_cb['novel_count']}/{novelty_cb['total']})" if novelty_cb["total"] > 0 else "  Clickbait    : N/A",
        f"  Non-clickbait: {novelty_ncb['novelty_ratio']:.2%}  ({novelty_ncb['novel_count']}/{novelty_ncb['total']})" if novelty_ncb["total"] > 0 else "  Non-clickbait: N/A",
        "",
        "--- 2. DIVERSITY (Keberagaman) ---",
        f"  Distinct-1   : {diversity_all.get('distinct_1', 'N/A')}",
        f"  Distinct-2   : {diversity_all.get('distinct_2', 'N/A')}",
        f"  Self-BLEU    : {diversity_all.get('self_bleu', 'N/A')} (lebih rendah = lebih beragam)",
        "",
        "--- 3. STYLE ACCURACY (Kualitas Gaya via Classifier) ---",
        f"  Akurasi gaya (semua) : {style_results.get('style_accuracy', 'N/A')}",
        f"  Clickbait accuracy   : {style_results.get('clickbait_accuracy', 'N/A')}",
        f"  Non-CB accuracy      : {style_results.get('non_clickbait_accuracy', 'N/A')}",
        "",
        "--- 4. PERPLEXITY (Kelancaran) ---",
        f"  Mean perplexity  : {ppl_results.get('mean_perplexity', 'N/A')} (lebih rendah = lebih lancar)",
        f"  Median perplexity: {ppl_results.get('median_perplexity', 'N/A')}",
        "",
        "--- JUDUL YANG DIHASILKAN ---",
    ]
    for r in generated:
        style_short = r.get("style", "?")[:2].upper()
        lines.append(f"  [{style_short}] {r.get('generated_text', '')}")
    lines.append("")
    lines.append("=" * 60)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Ringkasan teks disimpan di: {output_txt}", flush=True)

    # Cetak ringkasan ke terminal
    print("\n" + "\n".join(lines[: lines.index("--- JUDUL YANG DIHASILKAN ---") + 1]), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
