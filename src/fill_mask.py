"""
Step 3 - Contoh fill-mask untuk mengecek apakah model pretraining menebak kata yang masuk akal.

    python -m src.fill_mask checkpoints/pretrain/main/best.pt
    python -m src.fill_mask checkpoints/pretrain/main/best.pt "harga [MASK] naik menjelang lebaran."
"""

import sys

import torch

from config import CLS_ID, MASK_ID, SEP_ID
from src.data.cleaning import clean_title
from src.data.tokenizer import load_tokenizer
from src.model import load_model

EXAMPLES = [
    "presiden joko widodo meresmikan jalan [MASK] di jawa tengah.",
    "harga [MASK] naik menjelang hari raya idul fitri.",
    "polisi menangkap pelaku [MASK] di jakarta selatan.",
    "timnas indonesia [MASK] malaysia dengan skor 2-0.",
    "banjir [MASK] ribuan rumah warga di bekasi.",
]


def _encode(tokenizer, text):
    # [MASK] ditulis apa adanya; bagian lain dibersihkan seperti korpus (lowercase, dll.)
    parts = [clean_title(p) for p in text.split("[MASK]")]
    ids = [CLS_ID]
    for i, part in enumerate(parts):
        if i:
            ids.append(MASK_ID)
        if part:
            ids += tokenizer.encode(part, add_special_tokens=False).ids
    return ids + [SEP_ID]


@torch.no_grad()
def fill_mask(model, tokenizer, text, top_k=5):
    """Kembalikan daftar (per [MASK]) berisi top-k pasangan (token, probabilitas)."""
    device = next(model.parameters()).device
    ids = torch.tensor([_encode(tokenizer, text)], device=device)
    was_training = model.training
    model.eval()
    logits = model(ids)["logits"][0]
    model.train(was_training)
    results = []
    for pos in (ids[0] == MASK_ID).nonzero().flatten().tolist():
        probs = logits[pos].float().softmax(-1)
        top = probs.topk(top_k)
        results.append([(tokenizer.id_to_token(i), round(p, 4)) for p, i in zip(top.values.tolist(), top.indices.tolist())])
    return results


def format_result(text, result):
    return text + "\n" + "\n".join(
        f"    [MASK] {n + 1}: " + ", ".join(f"{tok} ({p:.2f})" for tok, p in cands) for n, cands in enumerate(result)
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit("pakai: python -m src.fill_mask <checkpoint.pt> [\"kalimat dengan [MASK]\"]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(sys.argv[1], device)
    tokenizer = load_tokenizer()
    for text in sys.argv[2:] or EXAMPLES:
        print(format_result(text, fill_mask(model, tokenizer, text)))


if __name__ == "__main__":
    main()
