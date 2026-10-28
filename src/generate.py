"""
Step 5 - Generasi Judul Berita dengan IndoMiniBERT (Mask-Predict).

Algoritma:
  1. Mulai dari template [CLS] [STYLE] [MASK] ... [MASK] [SEP] dengan panjang
     yang di-sample dari distribusi histogram judul di tokenizer_stats.json.
  2. Lakukan `generation_cfg.iterations` putaran mask-predict:
       a. Forward style_mlm -> logits untuk setiap posisi [MASK].
       b. Setiap posisi yang masih [MASK] digantikan token dengan probabilitas
          tertinggi (greedy) atau top-k sampling (temperature > 0).
       c. Pada iterasi selanjutnya, re-mask token dengan confidence terendah
          sebesar floor(ratio * n_mask_positions) token.
  3. Decode hasil akhir menjadi teks.

Cara jalankan (dari root repo):
    python -m src.generate --style clickbait --n 10
    python -m src.generate --style non-clickbait --n 5 --temp 0.8 --top-k 30
    python -m src.generate --style clickbait --n 8 --length 12

    Tanpa checkpoint (mode demo random):
    python -m src.generate --no-checkpoint --n 5

Output:
    reports/generated_titles.json    - hasil generasi dalam JSON
    reports/generated_titles.txt     - hasil generasi dalam plain text

Cara Generasi Manual (tanpa menjalankan script):
  Lihat docstring fungsi generate_titles_manual() di bawah untuk penjelasan
  langkah demi langkah cara melakukan mask-predict secara interaktif.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch

from config import (
    CB_ID, CLS_ID, MASK_ID, NCB_ID, PAD_ID, SEP_ID,
    LABEL_NAMES, SEED,
    generation_cfg, model_cfg, paths, style_cfg,
)
from src.data.tokenizer import load_tokenizer
from src.model import BertForMaskedLM


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_length_histogram() -> List[int]:
    """
    Muat histogram panjang judul (dalam token) dari tokenizer_stats.json.
    Index = jumlah token, nilai = frekuensi kemunculan.
    Digunakan untuk menyampel panjang judul yang realistis.
    """
    stats_path = paths.reports_dir / "tokenizer_stats.json"
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("title_length_histogram", [])
    # Fallback: distribusi seragam 6-18 token jika file tidak ada
    hist = [0] * 30
    for i in range(6, 19):
        hist[i] = 100
    return hist


def sample_title_length(histogram: List[int], min_len: int = 5, max_len: int = 20) -> int:
    """
    Sampel panjang judul (jumlah body token, di luar special tokens) dari histogram.
    min_len dan max_len membatasi panjang agar hasil tidak terlalu pendek/panjang.
    """
    weights = [
        (histogram[i] if i < len(histogram) else 0)
        for i in range(min_len, max_len + 1)
    ]
    total = sum(weights)
    if total == 0:
        return random.randint(min_len, max_len)
    r = random.random() * total
    cumulative = 0.0
    for offset, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return min_len + offset
    return max_len


# ---------------------------------------------------------------------------
# Panduan Generasi Manual
# ---------------------------------------------------------------------------

def generate_titles_manual():
    """
    PANDUAN GENERASI MANUAL - Mask-Predict Step by Step
    =====================================================
    Fungsi ini TIDAK dijalankan secara otomatis. Gunakan sebagai referensi
    untuk memahami cara kerja algoritma mask-predict dan cara menjalankannya
    secara interaktif di Python shell atau Jupyter Notebook.

    ================================================================
    LANGKAH 1: Import dan setup
    ================================================================

        import torch
        import json
        from config import CB_ID, NCB_ID, CLS_ID, SEP_ID, MASK_ID, paths, style_cfg
        from src.data.tokenizer import load_tokenizer
        from src.model import BertForMaskedLM

        device = torch.device("cpu")   # atau "cuda" jika ada GPU
        tok = load_tokenizer()
        model = BertForMaskedLM().to(device)

    ================================================================
    LANGKAH 2: Muat checkpoint style_mlm.pt
    ================================================================

        checkpoint_path = paths.checkpoints_dir / "style_mlm.pt"
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("Checkpoint dimuat!")

    ================================================================
    LANGKAH 3: Buat template awal (semua body token = MASK)
    ================================================================

        # Pilih gaya: CB_ID=5 (clickbait), NCB_ID=6 (non-clickbait)
        style_id = CB_ID      # clickbait
        body_len = 10         # panjang judul dalam token (di luar special tokens)
        max_len = style_cfg.max_len   # 48

        # Format: [CLS] [STYLE] [MASK] [MASK] ... [MASK] [SEP]
        token_ids = [CLS_ID, style_id] + [MASK_ID] * body_len + [SEP_ID]
        seq_len = len(token_ids)
        print("Template awal:", token_ids)

    ================================================================
    LANGKAH 4: Buat tensor input untuk model
    ================================================================

        input_ids = torch.full((1, max_len), 0, dtype=torch.long, device=device)
        attn_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)
        input_ids[0, :seq_len] = torch.tensor(token_ids, dtype=torch.long)
        attn_mask[0, :seq_len] = 1

    ================================================================
    LANGKAH 5: Jalankan mask-predict secara iteratif
    ================================================================

        n_iterations = 10
        body_positions = list(range(2, seq_len - 1))   # posisi body (setelah [CLS][STYLE])

        with torch.no_grad():
            for iteration in range(n_iterations):
                # 5a. Forward pass
                hidden_states = model.bert(input_ids, attn_mask)
                logits = model.mlm_head(hidden_states)    # shape: (1, max_len, vocab_size)

                # 5b. Isi setiap posisi yang masih MASK
                probs = torch.softmax(logits[0], dim=-1)  # shape: (max_len, vocab_size)
                new_ids = input_ids[0].clone()
                confidences = torch.zeros(max_len, device=device)

                for pos in body_positions:
                    # Ambil token dengan probabilitas tertinggi (greedy)
                    top_id = probs[pos].argmax().item()
                    new_ids[pos] = top_id
                    confidences[pos] = probs[pos, top_id].item()

                input_ids[0] = new_ids

                # 5c. Re-mask token dengan confidence terendah (kecuali iterasi terakhir)
                if iteration < n_iterations - 1:
                    remask_ratio = 1.0 - (iteration + 1) / n_iterations
                    n_remask = max(1, int(remask_ratio * len(body_positions)))
                    body_conf = sorted([(confidences[p].item(), p) for p in body_positions])
                    for _, pos in body_conf[:n_remask]:
                        input_ids[0, pos] = MASK_ID

                # Decode dan cetak status iterasi
                snap = [tok.id_to_token(input_ids[0, p].item()) for p in body_positions]
                print(f"Iter {iteration+1:2d}: {' '.join(t or '?' for t in snap)}")

    ================================================================
    LANGKAH 6: Decode hasil akhir menjadi teks
    ================================================================

        final_ids = input_ids[0, 2:seq_len-1].tolist()
        hasil_teks = tok.decode(final_ids, skip_special_tokens=True)
        print(f"Hasil generasi: {hasil_teks}")

    ================================================================
    LANGKAH 7 (Opsional): Top-k sampling dengan temperature
    ================================================================

        # Ganti langkah greedy di 5b dengan ini untuk hasil yang lebih beragam:
        temperature = 0.8
        top_k = 50

        scaled_logits = logits[0, pos] / temperature
        topk_vals, topk_idx = scaled_logits.topk(top_k)
        filtered = torch.full_like(scaled_logits, float("-inf"))
        filtered[topk_idx] = topk_vals
        chosen_id = torch.multinomial(filtered.softmax(-1), 1).item()
        new_ids[pos] = chosen_id
        confidences[pos] = filtered.softmax(-1)[chosen_id].item()

    ================================================================
    CATATAN PENTING
    ================================================================
    - Semakin banyak iterasi, semakin koheren hasilnya (tapi lambat).
    - Temperature < 1.0 -> lebih deterministik; > 1.0 -> lebih beragam/acak.
    - Top-k = 1 setara dengan greedy decoding.
    - Judul yang baik biasanya memiliki avg_confidence > 0.3.
    """
    pass   # Fungsi ini hanya dokumentasi


# ---------------------------------------------------------------------------
# Fungsi pembersihan & utilitas tanda baca
# ---------------------------------------------------------------------------

def clean_generated_headline(text: str) -> str:
    """
    Merapikan teks judul hasil decode WordPiece:
    - Menghapus tanda baca berulang (misal: ',,,,' -> ',')
    - Menghapus spasi yang mendahului tanda baca (misal: 'norma ,' -> 'norma,')
    - Menghapus tanda baca di awal kalimat
    - Menghapus tanda koma / titik dua gantung di akhir kalimat
    - Merapikan spasi ganda
    """
    if not text:
        return ""
    # Spasi sebelum tanda baca: "kata ," -> "kata,"
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    # Tanda baca berulang berurutan: ",,,," -> ",", ".. " -> "."
    text = re.sub(r"([.,!?:;])\s*\1+", r"\1", text)
    # Kombinasi koma diikuti tanda baca akhir: ",." -> "."
    text = re.sub(r",\s*([.!?])", r"\1", text)
    # Tanda baca di awal kalimat: ", presiden ..." -> "presiden ..."
    text = re.sub(r"^[\s,.\-:;]+", "", text)
    # Tanda koma / titik dua gantung di akhir kalimat
    text = re.sub(r"[\s,;:\-]+$", "", text)
    # Rapikan spasi ganda
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_punctuation_token_ids(tokenizer) -> Set[int]:
    """Ambil seluruh ID token yang merupakan tanda baca dari vocabulary tokenizer."""
    import string
    punct_chars = set(string.punctuation) | {"...", "--", "“", "”", "‘", "’"}
    punct_ids = set()
    for tid in range(tokenizer.get_vocab_size()):
        tok = tokenizer.id_to_token(tid)
        if tok:
            clean_tok = tok.replace("##", "").strip()
            if clean_tok and all(ch in punct_chars for ch in clean_tok):
                punct_ids.add(tid)
    return punct_ids


# ---------------------------------------------------------------------------
# Fungsi generasi inti
# ---------------------------------------------------------------------------

def mask_predict_generate(
    model: BertForMaskedLM,
    tokenizer,
    style_id: int,
    body_len: int,
    device: torch.device,
    n_iterations: int = 10,
    top_k: int = 50,
    temperature: float = 1.0,
    repetition_penalty: float = 1.3,
    no_consecutive_punct: bool = True,
    punct_token_ids: Optional[Set[int]] = None,
) -> dict:
    """
    Hasilkan satu judul menggunakan algoritma mask-predict dengan repetition penalty
    dan aturan larangan tanda baca berurutan.

    Args:
        model               : BertForMaskedLM yang sudah di-load dan .eval()
        tokenizer           : tokenizers.Tokenizer
        style_id            : CB_ID (5) untuk clickbait, NCB_ID (6) untuk non-clickbait
        body_len            : panjang judul dalam token (di luar [CLS], [STYLE], [SEP])
        device              : torch.device
        n_iterations        : jumlah putaran mask-predict
        top_k               : filter top-k kandidat token saat sampling (1 = greedy)
        temperature         : suhu sampling; < 1.0 lebih confident; > 1.0 lebih acak
        repetition_penalty  : penalti probabilitas untuk token yang sudah muncul (> 1.0)
        no_consecutive_punct: larang tanda baca berturut-turut (misal koma ganda ,,)
        punct_token_ids     : set ID token tanda baca (diambil otomatis jika None)

    Returns:
        dict berisi generated_text, token_ids, iterations_snapshots, avg_confidence
    """
    max_len = style_cfg.max_len
    seq_len = min(2 + body_len + 1, max_len)
    body_len = seq_len - 3

    if punct_token_ids is None:
        punct_token_ids = get_punctuation_token_ids(tokenizer)

    comma_id = tokenizer.token_to_id(",")
    colon_id = tokenizer.token_to_id(":")

    # Template: [CLS] [STYLE] [MASK]...body...[MASK] [SEP]
    token_ids = [CLS_ID, style_id] + [MASK_ID] * body_len + [SEP_ID]

    input_ids = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
    attn_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)
    input_ids[0, :seq_len] = torch.tensor(token_ids, dtype=torch.long, device=device)
    attn_mask[0, :seq_len] = 1

    body_positions = list(range(2, seq_len - 1))
    snapshots = []

    with torch.no_grad():
        for iteration in range(n_iterations):
            # Forward pass
            hidden_states = model.bert(input_ids, attn_mask)
            logits = model.mlm_head(hidden_states)

            new_ids = input_ids[0].clone()
            confidences = torch.zeros(max_len, device=device)

            for pos in body_positions:
                cur_logits = logits[0, pos].clone()

                # 1. Larang Special Tokens (0..NCB_ID) muncul di badan judul
                cur_logits[:NCB_ID + 1] = float("-inf")

                # 2. Aturan Larangan Tanda Baca Berurutan & Posisi Ekstrem
                if no_consecutive_punct:
                    # Posisi awal (kata pertama judul) tidak boleh tanda baca
                    if pos == body_positions[0]:
                        for p_id in punct_token_ids:
                            cur_logits[p_id] = float("-inf")

                    # Posisi setelah tanda baca tidak boleh tanda baca lagi (hindari ,, atau ,.)
                    if pos > body_positions[0]:
                        prev_tok = new_ids[pos - 1].item()
                        if prev_tok in punct_token_ids:
                            for p_id in punct_token_ids:
                                cur_logits[p_id] = float("-inf")

                    # Posisi terakhir sebelum [SEP] tidak boleh koma atau titik dua gantung
                    if pos == body_positions[-1]:
                        if comma_id is not None:
                            cur_logits[comma_id] = float("-inf")
                        if colon_id is not None:
                            cur_logits[colon_id] = float("-inf")

                # 3. Larang token yang sama persis berturut-turut (misal: "norma norma")
                if pos > body_positions[0]:
                    prev_tok = new_ids[pos - 1].item()
                    if prev_tok not in (PAD_ID, MASK_ID):
                        cur_logits[prev_tok] = float("-inf")

                # 4. Batasi tanda koma (maksimal 1 koma di seluruh judul)
                if comma_id is not None:
                    comma_count = sum(
                        1 for p in body_positions if p != pos and new_ids[p].item() == comma_id
                    )
                    if comma_count >= 1:
                        cur_logits[comma_id] = float("-inf")

                # 5. Repetition Penalty untuk token yang sudah muncul di kalimat
                if repetition_penalty > 1.0:
                    for p in body_positions:
                        if p != pos:
                            tok_in_seq = new_ids[p].item()
                            if tok_in_seq > NCB_ID and tok_in_seq != MASK_ID:
                                if cur_logits[tok_in_seq] > 0:
                                    cur_logits[tok_in_seq] /= repetition_penalty
                                else:
                                    cur_logits[tok_in_seq] *= repetition_penalty

                # 6. Sampling (Greedy atau Top-k dengan temperature)
                if temperature <= 0.0 or top_k <= 1:
                    chosen_id = int(cur_logits.argmax().item())
                else:
                    scaled = cur_logits / max(temperature, 1e-5)
                    topk_vals, topk_idx = scaled.topk(min(top_k, model_cfg.vocab_size))
                    filtered = torch.full_like(scaled, float("-inf"))
                    filtered[topk_idx] = topk_vals
                    chosen_id = int(torch.multinomial(filtered.softmax(-1), 1).item())

                new_ids[pos] = chosen_id
                probs = cur_logits.softmax(-1)
                confidences[pos] = probs[chosen_id].item()

            input_ids[0] = new_ids

            # Snapshot teks saat ini
            snap_ids = input_ids[0, 2:seq_len - 1].tolist()
            snap_tokens = [tokenizer.id_to_token(t) or "?" for t in snap_ids]
            raw_snap_text = tokenizer.decode(snap_ids, skip_special_tokens=True)
            snapshots.append({
                "iteration": iteration + 1,
                "tokens": snap_tokens,
                "text": clean_generated_headline(raw_snap_text),
            })

            # Re-mask token berconfidence rendah (kecuali iterasi terakhir)
            if iteration < n_iterations - 1:
                remask_ratio = 1.0 - (iteration + 1) / n_iterations
                n_remask = max(1, int(math.ceil(remask_ratio * len(body_positions))))
                
                # Prioritaskan me-remask tanda baca berurutan jika ada yang lolos
                body_conf = []
                for p in body_positions:
                    c = confidences[p].item()
                    if p > body_positions[0] and new_ids[p].item() in punct_token_ids and new_ids[p - 1].item() in punct_token_ids:
                        c = -1.0  # paksa remask
                    body_conf.append((c, p))

                body_conf.sort()
                for _, pos in body_conf[:n_remask]:
                    input_ids[0, pos] = MASK_ID

    final_ids = input_ids[0, 2:seq_len - 1].tolist()
    raw_final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    final_text = clean_generated_headline(raw_final_text)
    avg_confidence = float(
        sum(confidences[p].item() for p in body_positions) / max(len(body_positions), 1)
    )

    return {
        "style": LABEL_NAMES[1 if style_id == CB_ID else 0],
        "body_len_tokens": body_len,
        "generated_text": final_text,
        "token_ids": final_ids,
        "avg_confidence": round(avg_confidence, 4),
        "iterations_snapshots": snapshots,
    }


def generate_titles(
    model: BertForMaskedLM,
    tokenizer,
    style: str,
    n: int,
    device: torch.device,
    histogram: List[int],
    n_iterations: int = 10,
    top_k: int = 50,
    temperature: float = 1.0,
    repetition_penalty: float = 1.3,
    no_consecutive_punct: bool = True,
    fixed_length: Optional[int] = None,
) -> List[dict]:
    """
    Hasilkan n judul dengan gaya tertentu.

    Args:
        style               : "clickbait" atau "non-clickbait"
        n                   : jumlah judul yang dihasilkan
        fixed_length        : jika tidak None, pakai panjang tetap ini (dalam body token)
        repetition_penalty : penalti repetisi token (> 1.0)
        no_consecutive_punct: cegah tanda baca berturut-turut
    """
    style_id = CB_ID if style.lower() in ("clickbait", "cb") else NCB_ID
    punct_token_ids = get_punctuation_token_ids(tokenizer)
    results = []

    for i in range(n):
        body_len = fixed_length if fixed_length is not None else sample_title_length(histogram)
        result = mask_predict_generate(
            model=model,
            tokenizer=tokenizer,
            style_id=style_id,
            body_len=body_len,
            device=device,
            n_iterations=n_iterations,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_consecutive_punct=no_consecutive_punct,
            punct_token_ids=punct_token_ids,
        )
        result["idx"] = i + 1
        results.append(result)
        print(
            f"  [{i+1:2d}/{n}] [{result['style'][:2].upper()}] "
            f"(len={body_len}, conf={result['avg_confidence']:.3f}) "
            f"{result['generated_text']}",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Generasi judul berita Indonesia dengan mask-predict"
    )
    parser.add_argument(
        "--style", choices=["clickbait", "non-clickbait", "cb", "ncb", "both"],
        default="both",
        help="Gaya judul (default: both -> hasilkan clickbait dan non-clickbait)",
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Jumlah judul per gaya (default: 10)")
    parser.add_argument("--iterations", type=int, default=generation_cfg.iterations,
                        help=f"Jumlah iterasi mask-predict (default: {generation_cfg.iterations})")
    parser.add_argument("--top-k", type=int, default=generation_cfg.top_k,
                        help=f"Top-k kandidat token (default: {generation_cfg.top_k})")
    parser.add_argument("--temp", type=float, default=generation_cfg.temperature,
                        help=f"Temperature sampling (default: {generation_cfg.temperature})")
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="Penalti repetisi token yang sudah ada (default: 1.3, 1.0 = nonaktif)")
    parser.add_argument("--allow-consecutive-punct", action="store_true",
                        help="Izinkan tanda baca berurutan (default: larang tanda baca berurutan)")
    parser.add_argument("--length", type=int, default=None,
                        help="Panjang tetap body token (default: disampling dari histogram)")
    parser.add_argument("--checkpoint",
                        default=str(paths.checkpoints_dir / "style_mlm.pt"))
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Pakai bobot random tanpa checkpoint (mode demo)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json",
                        default=str(paths.reports_dir / "generated_titles.json"))
    parser.add_argument("--output-txt",
                        default=str(paths.reports_dir / "generated_titles.txt"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia.")

    tokenizer = load_tokenizer()
    model = BertForMaskedLM().to(device)

    if args.no_checkpoint:
        print("Mode demo: model dengan bobot random (--no-checkpoint).", flush=True)
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {ckpt_path}\n"
                "Solusi:\n"
                "  1. Jalankan src.finetune_style terlebih dahulu.\n"
                "  2. Atau gunakan --no-checkpoint untuk demo tanpa checkpoint.\n"
                "  3. Atau tentukan path dengan --checkpoint <path>"
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Checkpoint dimuat dari: {ckpt_path}", flush=True)

    model.eval()
    histogram = load_length_histogram()

    styles_to_gen = (
        ["clickbait", "non-clickbait"] if args.style == "both"
        else ["clickbait"] if args.style in ("clickbait", "cb")
        else ["non-clickbait"]
    )

    print(
        f"\nGenerasi {args.n} judul per gaya | iter={args.iterations} "
        f"top_k={args.top_k} temp={args.temp} rep_pen={args.repetition_penalty} "
        f"no_repeat_punct={not args.allow_consecutive_punct} device={device}\n",
        flush=True,
    )

    all_results = []
    for style in styles_to_gen:
        print(f"--- Gaya: {style.upper()} ---", flush=True)
        results = generate_titles(
            model=model, tokenizer=tokenizer, style=style, n=args.n,
            device=device, histogram=histogram, n_iterations=args.iterations,
            top_k=args.top_k, temperature=args.temp,
            repetition_penalty=args.repetition_penalty,
            no_consecutive_punct=not args.allow_consecutive_punct,
            fixed_length=args.length,
        )
        all_results.extend(results)
        print()

    # Simpan output JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "style_mlm",
        "checkpoint": args.checkpoint if not args.no_checkpoint else "random_weights",
        "device": str(device),
        "n_per_style": args.n,
        "iterations": args.iterations,
        "top_k": args.top_k,
        "temperature": args.temp,
        "repetition_penalty": args.repetition_penalty,
        "no_consecutive_punct": not args.allow_consecutive_punct,
        "fixed_length": args.length,
        "seed": args.seed,
        "generated": all_results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Output JSON: {output_json}", flush=True)

    # Simpan output TXT
    output_txt = Path(args.output_txt)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  HASIL GENERASI JUDUL - IndoMiniBERT Step 5\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"[{r['style'].upper()[:2]}] {r['generated_text']}\n")
        f.write(f"\nTotal judul: {len(all_results)}\n")
    print(f"Output TXT: {output_txt}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  RANGKUMAN HASIL GENERASI", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        print(f"  [{r['style'][:2].upper()}] {r['generated_text']}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
