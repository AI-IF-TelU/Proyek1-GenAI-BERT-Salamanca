"""
Step 1 - Tokenizer WordPiece ~16.000 token, dilatih dari korpus raw CLICK-ID.

Jalankan dari root repo (setelah src.data.prepare):
    python -m src.data.tokenizer

Token khusus [PAD] [UNK] [CLS] [SEP] [MASK] [CB] [NCB] dimasukkan saat training
sehingga ID-nya tetap 0..6 (lihat config.py). Tokenizer dilatih hanya dari
pretrain_train.txt, yang sudah bersih dari judul split test.

Pemakaian di modul lain:
    from src.data.tokenizer import load_tokenizer
    tok = load_tokenizer()
    ids = tok.encode("presiden resmikan jalan tol", add_special_tokens=False).ids
"""

import json

import numpy as np
import pandas as pd
from tokenizers import BertWordPieceTokenizer, Tokenizer

from config import SPECIAL_TOKENS, paths, style_cfg, tokenizer_cfg


def train_tokenizer() -> Tokenizer:
    corpus = paths.pretrain_split["train"]
    if not corpus.exists():
        raise FileNotFoundError(f"{corpus} belum ada, jalankan dulu: python -m src.data.prepare")

    tok = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,          # bahasa Indonesia jarang beraksen, tidak perlu
        lowercase=tokenizer_cfg.lowercase,
    )
    tok.train(
        files=[str(corpus)],
        vocab_size=tokenizer_cfg.vocab_size,
        min_frequency=tokenizer_cfg.min_frequency,
        special_tokens=tokenizer_cfg.special_tokens,
        limit_alphabet=1000,
        wordpieces_prefix=tokenizer_cfg.wordpiece_prefix,
        show_progress=True,
    )
    paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(paths.tokenizer_file))
    tok.save_model(str(paths.tokenizer_dir))     # vocab.txt, untuk dibaca manusia
    return load_tokenizer()


def load_tokenizer() -> Tokenizer:
    """Load tokenizer dan pastikan ID token khusus sesuai config.py."""
    if not paths.tokenizer_file.exists():
        raise FileNotFoundError(f"{paths.tokenizer_file} belum ada, jalankan: python -m src.data.tokenizer")
    tok = Tokenizer.from_file(str(paths.tokenizer_file))
    tok.no_padding()
    tok.no_truncation()
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tok.token_to_id(token)
        if actual != expected_id:
            raise ValueError(f"ID {token} = {actual}, seharusnya {expected_id}. Latih ulang tokenizer.")
    return tok


def tokenizer_stats(tok: Tokenizer) -> dict:
    """Statistik panjang token judul, dipakai untuk memilih max_len=48 dan oleh Step 5."""
    train = pd.read_csv(paths.title_split["train"])
    lengths = np.array([len(e.ids) for e in tok.encode_batch(train["text"].tolist(), add_special_tokens=False)])
    # [CLS] + token gaya + judul + [SEP]
    budget = style_cfg.max_len - 3
    hist = np.bincount(lengths)

    unk_id = tok.token_to_id("[UNK]")
    sample = train["text"].tolist()
    all_ids = [i for e in tok.encode_batch(sample, add_special_tokens=False) for i in e.ids]

    examples = {}
    for text in sample[:5]:
        examples[text] = tok.encode(text, add_special_tokens=False).tokens

    return {
        "vocab_size": tok.get_vocab_size(),
        "title_tokens": {
            "mean": round(float(lengths.mean()), 2),
            "p50": int(np.percentile(lengths, 50)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            "truncated_at_max_len": int((lengths > budget).sum()),
            "max_len": style_cfg.max_len,
        },
        # distribusi panjang judul (index = jumlah token), dipakai Step 5 untuk sampling panjang
        "title_length_histogram": hist.tolist(),
        "title_unk_rate": round(all_ids.count(unk_id) / max(len(all_ids), 1), 5),
        "tokenization_examples": examples,
    }


def main():
    tok = train_tokenizer()
    stats = tokenizer_stats(tok)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out = paths.reports_dir / "tokenizer_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "title_length_histogram"}, indent=2, ensure_ascii=False))
    print(f"\nTokenizer disimpan di {paths.tokenizer_dir}, statistik di {out}")


if __name__ == "__main__":
    main()
