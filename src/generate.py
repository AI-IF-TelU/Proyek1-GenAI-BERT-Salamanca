"""
Step 5 - Generasi Judul Berita dengan IndoMiniBERT (Mask-Predict).

Algoritma Mask-Predict Disempurnakan:
  1. Mulai dari template [CLS] [STYLE] [STARTER/PROMPT] [MASK] ... [MASK] [SEP]
     dengan panjang yang di-sample dari distribusi histogram judul.
  2. Inisialisasi Sekuensial (Pass 0): menebak posisi [MASK] secara berurutan
     agar kata saling memperhatikan konteks (tidak independen).
  3. Larangan Subword & Punctuation:
     - Melarang subword kelanjutan ('##...') agar tidak menghasilkan kata buatan
       seperti 'budiocoran', 'dipahamiana', 'norma2onc', dll.
     - Melarang tanda baca beruntun (,,) dan membatasi maksimal 1 koma per judul.
  4. Repetition Penalty:
     - Mencegah kata yang sama diulang dalam satu judul.
     - Mencegah kata frekuensi tinggi mendominasi semua judul dalam satu sesi.
  5. Refinement Mask-Predict:
     - Melakukan re-masking pada token dengan confidence terendah dan menebak ulang
       dengan konteks dua arah.
  6. Decode & Clean: merapikan spasi dan tanda baca akhir.

Cara jalankan (dari root repo):
    python -m src.generate --style clickbait --n 5
    python -m src.generate --style non-clickbait --n 5
    python -m src.generate --style both --n 5 --temp 0.85 --top-k 150

    Dengan kata kunci / prompt (Mode 1 & 2):
    python -m src.generate --prompt "presiden jokowi resmikan [MASK] [MASK] di jakarta"
    python -m src.generate --keywords "persib, juara" --style clickbait

Output:
    reports/generated_titles.json    - hasil generasi dalam JSON
    reports/generated_titles.txt     - hasil generasi dalam plain text
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


def sample_title_length(histogram: List[int], min_len: int = 6, max_len: int = 16) -> int:
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
    Fungsi dokumentasi langkah demi langkah cara melakukan generasi mask-predict.
    """
    pass


# ---------------------------------------------------------------------------
# Pembersihan & Utilitas Token
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
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"([.,!?:;])\s*\1+", r"\1", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"^[\s,.\-:;]+", "", text)
    text = re.sub(r"[\s,;:\-]+$", "", text)
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


def get_banned_token_ids(tokenizer) -> Set[int]:
    """
    Ambil ID token yang dilarang muncul dalam badan judul:
    1. Special tokens ([PAD], [UNK], [CLS], [SEP], [MASK], [CB], [NCB])
    2. Subword kelanjutan (diawali '##') agar tidak menghasilkan kata Frankenstein
       seperti 'budiocoran', 'dipahamiana', 'otakorasi', 'norma2onc', dll.
    3. Noise sisa scraping portal (tanggal 'agu', serpihan 'seper', 'kej', 'tri', dll.)
    4. Token satu huruf acak yang bukan kata tugas baku.
    """
    banned = set()
    noise_words = {
        "agu", "seper", "kej", "tri", "pg", "mrt", "gemp", "na", "bn",
        "onc", "baca", "erkan", "dian", "torat", "annya", "punya",
    }
    for tid in range(tokenizer.get_vocab_size()):
        if tid <= NCB_ID:
            banned.add(tid)
            continue
        tok = tokenizer.id_to_token(tid)
        if not tok:
            banned.add(tid)
            continue
        if tok.startswith("##"):
            banned.add(tid)
            continue
        if tok in noise_words:
            banned.add(tid)
            continue
        if len(tok) <= 1 and tok not in ("di", "ke"):
            banned.add(tid)
            continue

    return banned


STYLE_STARTERS = {
    "clickbait": [
        "viral", "ternyata", "begini", "fakta", "rahasia", "heboh",
        "nekat", "curhat", "terungkap", "alasan", "bikin", "penasaran",
        "kaget", "reaksi", "misteri", "pengakuan",
    ],
    "non-clickbait": [
        "pemerintah", "presiden", "polisi", "menteri", "kpk", "dpr",
        "bmkg", "kasus", "indonesia", "jadwal", "laga", "hasil",
        "pemprov", "jakarta", "kemenkes",
    ],
}


# ---------------------------------------------------------------------------
# Fungsi Generasi Inti
# ---------------------------------------------------------------------------

def mask_predict_generate(
    model: BertForMaskedLM,
    tokenizer,
    style_id: int,
    body_len: int,
    device: torch.device,
    prompt: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    n_iterations: int = 5,
    top_k: int = 150,
    temperature: float = 0.85,
    repetition_penalty: float = 1.6,
    no_consecutive_punct: bool = True,
    banned_token_ids: Optional[Set[int]] = None,
    punct_token_ids: Optional[Set[int]] = None,
    batch_used_tokens: Optional[Set[int]] = None,
) -> dict:
    """
    Hasilkan satu judul menggunakan algoritma mask-predict yang disempurnakan:
    1. Melarang subword kelanjutan ('##...') agar tidak menghasilkan kata aneh.
    2. Melarang tanda baca beruntun dan membatasi maksimal 1 tanda koma.
    3. Inisialisasi sekuensial (autoregressive pass) agar kata saling berkorelasi.
    4. Refinement bertahap via mask-predict pada token ber-confidence terendah.
    5. Mendukung prompt awal / fill-mask / kata kunci jika disediakan pengguna.
    """
    max_len = style_cfg.max_len
    seq_len = min(2 + body_len + 1, max_len)
    body_len = seq_len - 3

    if banned_token_ids is None:
        banned_token_ids = get_banned_token_ids(tokenizer)
    if punct_token_ids is None:
        punct_token_ids = get_punctuation_token_ids(tokenizer)
    if batch_used_tokens is None:
        batch_used_tokens = set()

    comma_id = tokenizer.token_to_id(",")
    colon_id = tokenizer.token_to_id(":")
    style_name = LABEL_NAMES[1 if style_id == CB_ID else 0]

    input_ids = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
    attn_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)

    fixed_positions = set()

    if prompt:
        # Mode 1: Fill-mask / Custom Prompt
        encoded = tokenizer.encode(prompt.lower(), add_special_tokens=False)
        p_ids = encoded.ids
        body_len = max(len(p_ids), body_len)
        seq_len = min(2 + body_len + 1, max_len)
        token_list = [CLS_ID, style_id] + p_ids[:seq_len - 3]
        while len(token_list) < seq_len - 1:
            token_list.append(MASK_ID)
        token_list.append(SEP_ID)
        for idx_p, tid in enumerate(token_list[2:-1], start=2):
            if tid != MASK_ID:
                fixed_positions.add(idx_p)
        input_ids[0, :seq_len] = torch.tensor(token_list, dtype=torch.long, device=device)
    elif keywords:
        # Mode 2: Kata kunci menjadi judul
        kw_tokens = []
        for kw in keywords:
            kw_tokens.extend(tokenizer.encode(kw.lower(), add_special_tokens=False).ids)
        token_list = [CLS_ID, style_id] + [MASK_ID] * body_len + [SEP_ID]
        for idx_k, kid in enumerate(kw_tokens[:body_len]):
            token_list[2 + idx_k] = kid
            fixed_positions.add(2 + idx_k)
        input_ids[0, :seq_len] = torch.tensor(token_list, dtype=torch.long, device=device)
    else:
        # Mode 3: Generasi Bebas terpandu (pilih starter alami sesuai gaya)
        starter_candidates = STYLE_STARTERS.get(style_name, ["indonesia"])
        starter_word = random.choice(starter_candidates)
        s_id = tokenizer.token_to_id(starter_word)
        if s_id is not None:
            token_list = [CLS_ID, style_id, s_id] + [MASK_ID] * (body_len - 1) + [SEP_ID]
            fixed_positions.add(2)
        else:
            token_list = [CLS_ID, style_id] + [MASK_ID] * body_len + [SEP_ID]
        input_ids[0, :seq_len] = torch.tensor(token_list, dtype=torch.long, device=device)

    attn_mask[0, :seq_len] = 1
    body_positions = list(range(2, seq_len - 1))
    current_tokens = {input_ids[0, p].item() for p in fixed_positions}

    def _sample_from_logits(logits_pos: torch.Tensor, pos: int, current_seq: Set[int]) -> int:
        cur_l = logits_pos.clone()

        # 1. Larang token terlarang (special token, ## subword, noise)
        for b_id in banned_token_ids:
            cur_l[b_id] = float("-inf")

        # 2. Aturan tanda baca
        if no_consecutive_punct:
            if pos == body_positions[0]:
                for p_id in punct_token_ids:
                    cur_l[p_id] = float("-inf")

            if pos > body_positions[0]:
                prev_tok = input_ids[0, pos - 1].item()
                if prev_tok in punct_token_ids:
                    for p_id in punct_token_ids:
                        cur_l[p_id] = float("-inf")

            if pos == body_positions[-1]:
                if comma_id is not None:
                    cur_l[comma_id] = float("-inf")
                if colon_id is not None:
                    cur_l[colon_id] = float("-inf")

        # Batasi koma max 1 dalam satu judul
        if comma_id is not None and comma_id in current_seq:
            cur_l[comma_id] = float("-inf")

        # 3. Larang pengulangan token yang sama berturut-turut
        if pos > body_positions[0]:
            prev_tok = input_ids[0, pos - 1].item()
            if prev_tok not in (PAD_ID, MASK_ID):
                cur_l[prev_tok] = float("-inf")

        # 4. Repetition Penalty dalam satu judul
        if repetition_penalty > 1.0:
            for t in current_seq:
                if t > NCB_ID and t != MASK_ID:
                    if cur_l[t] > 0:
                        cur_l[t] /= repetition_penalty
                    else:
                        cur_l[t] *= repetition_penalty

        # 5. Penalti lembut untuk token yang sering muncul di batch sebelumnya
        for t in batch_used_tokens:
            if t > NCB_ID and t != MASK_ID:
                if cur_l[t] > 0:
                    cur_l[t] /= 1.25
                else:
                    cur_l[t] *= 1.25

        # 6. Sampling (Greedy vs Top-K Temperature)
        if temperature <= 0.0 or top_k <= 1:
            return int(cur_l.argmax().item())

        scaled = cur_l / max(temperature, 1e-5)
        topk_vals, topk_idx = scaled.topk(min(top_k, model_cfg.vocab_size))
        filtered = torch.full_like(scaled, float("-inf"))
        filtered[topk_idx] = topk_vals
        return int(torch.multinomial(filtered.softmax(-1), 1).item())

    # -----------------------------------------------------------------------
    # Fase 1: Sequential Autoregressive Initialization
    # -----------------------------------------------------------------------
    with torch.no_grad():
        for pos in body_positions:
            if pos in fixed_positions:
                continue
            out = model(input_ids, attn_mask)
            chosen_id = _sample_from_logits(out[0, pos], pos, current_tokens)
            input_ids[0, pos] = chosen_id
            current_tokens.add(chosen_id)

    # -----------------------------------------------------------------------
    # Fase 2: Mask-Predict Refinement
    # -----------------------------------------------------------------------
    snapshots = []
    with torch.no_grad():
        for it in range(n_iterations):
            out = model(input_ids, attn_mask)
            confidences = torch.zeros(max_len, device=device)

            for pos in body_positions:
                prob = out[0, pos].softmax(-1)[input_ids[0, pos].item()].item()
                confidences[pos] = prob

            snap_ids = input_ids[0, 2:seq_len - 1].tolist()
            raw_text = tokenizer.decode(snap_ids, skip_special_tokens=True)
            snapshots.append({
                "iteration": it + 1,
                "text": clean_generated_headline(raw_text),
            })

            if it < n_iterations - 1:
                eligible_remask = [p for p in body_positions if p not in fixed_positions]
                if not eligible_remask:
                    break
                n_remask = max(1, min(len(eligible_remask), int(math.ceil(0.25 * len(body_positions)))))
                sorted_by_conf = sorted([(confidences[p].item(), p) for p in eligible_remask])
                remask_positions = [p for _, p in sorted_by_conf[:n_remask]]

                for rpos in remask_positions:
                    input_ids[0, rpos] = MASK_ID

                out_refine = model(input_ids, attn_mask)
                for rpos in remask_positions:
                    chosen_id = _sample_from_logits(out_refine[0, rpos], rpos, current_tokens)
                    input_ids[0, rpos] = chosen_id
                    current_tokens.add(chosen_id)

    final_ids = input_ids[0, 2:seq_len - 1].tolist()
    raw_final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
    final_text = clean_generated_headline(raw_final_text)
    avg_conf = float(sum(confidences[p].item() for p in body_positions) / max(len(body_positions), 1))

    return {
        "style": style_name,
        "body_len_tokens": body_len,
        "generated_text": final_text,
        "token_ids": final_ids,
        "avg_confidence": round(avg_conf, 4),
        "iterations_snapshots": snapshots,
    }


def generate_titles(
    model: BertForMaskedLM,
    tokenizer,
    style: str,
    n: int,
    device: torch.device,
    histogram: List[int],
    prompt: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    n_iterations: int = 5,
    top_k: int = 150,
    temperature: float = 0.85,
    repetition_penalty: float = 1.6,
    no_consecutive_punct: bool = True,
    fixed_length: Optional[int] = None,
) -> List[dict]:
    """
    Hasilkan n judul dengan gaya tertentu.
    """
    style_id = CB_ID if style.lower() in ("clickbait", "cb") else NCB_ID
    banned_token_ids = get_banned_token_ids(tokenizer)
    punct_token_ids = get_punctuation_token_ids(tokenizer)
    batch_used_tokens = set()
    results = []

    for i in range(n):
        body_len = fixed_length if fixed_length is not None else sample_title_length(histogram)
        result = mask_predict_generate(
            model=model,
            tokenizer=tokenizer,
            style_id=style_id,
            body_len=body_len,
            device=device,
            prompt=prompt,
            keywords=keywords,
            n_iterations=n_iterations,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_consecutive_punct=no_consecutive_punct,
            banned_token_ids=banned_token_ids,
            punct_token_ids=punct_token_ids,
            batch_used_tokens=batch_used_tokens,
        )
        batch_used_tokens.update(result["token_ids"])
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
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt awal judul / template fill-mask (opsional)")
    parser.add_argument("--keywords", type=str, default=None,
                        help="Kata kunci dipisahkan koma, misal: 'persib, juara' (opsional)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Jumlah iterasi mask-predict (default: 5)")
    parser.add_argument("--top-k", type=int, default=150,
                        help="Top-k kandidat token sampling (default: 150)")
    parser.add_argument("--temp", type=float, default=0.85,
                        help="Temperature sampling (default: 0.85)")
    parser.add_argument("--repetition-penalty", type=float, default=1.6,
                        help="Penalti repetisi token (default: 1.6)")
    parser.add_argument("--allow-consecutive-punct", action="store_true",
                        help="Izinkan tanda baca berurutan (default: dilarang)")
    parser.add_argument("--length", type=int, default=None,
                        help="Panjang tetap body token (default: sampling dari histogram)")
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

    kw_list = [k.strip() for k in args.keywords.split(",")] if args.keywords else None

    print(
        f"\nGenerasi {args.n} judul per gaya | iter={args.iterations} "
        f"top_k={args.top_k} temp={args.temp} rep_pen={args.repetition_penalty} "
        f"device={device}\n",
        flush=True,
    )

    all_results = []
    for style in styles_to_gen:
        print(f"--- Gaya: {style.upper()} ---", flush=True)
        results = generate_titles(
            model=model, tokenizer=tokenizer, style=style, n=args.n,
            device=device, histogram=histogram, prompt=args.prompt,
            keywords=kw_list, n_iterations=args.iterations,
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
        "prompt": args.prompt,
        "keywords": kw_list,
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
