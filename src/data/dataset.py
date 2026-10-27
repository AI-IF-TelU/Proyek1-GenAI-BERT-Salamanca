"""
Step 1 - Dataset class untuk pretraining MLM dan judul berlabel.

PretrainDataset  -> input_ids, attention_mask           shape (128,)
    Kalimat dari satu artikel dipadatkan (packing) sampai 126 token, lalu dibungkus
    [CLS] ... [SEP] dan di-pad ke 128. Packing tidak melewati batas artikel.
    Masking TIDAK dilakukan di sini; itu tugas data collator di Step 3.

TitleDataset     -> input_ids, attention_mask, labels   shape (48,)
    with_style=True  : [CLS] [CB]/[NCB] judul [SEP]  (MLM bergaya, Step 4)
    with_style=False : [CLS] judul [SEP]             (classifier clickbait, Step 4)

Contoh:
    from src.data.dataset import build_pretrain_loader, build_title_loader
    loader = build_pretrain_loader("train", batch_size=128)
    batch = next(iter(loader))     # batch["input_ids"].shape == (128, 128)
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import CLS_ID, LABEL2STYLE_ID, PAD_ID, SEP_ID, paths, pretrain_cfg, style_cfg
from src.data.tokenizer import load_tokenizer


def _pad(ids: list, max_len: int) -> tuple:
    n = len(ids)
    input_ids = torch.full((max_len,), PAD_ID, dtype=torch.long)
    input_ids[:n] = torch.tensor(ids, dtype=torch.long)
    attention_mask = torch.zeros(max_len, dtype=torch.long)
    attention_mask[:n] = 1
    return input_ids, attention_mask


def read_docs(path) -> list:
    """Baca file korpus: satu kalimat per baris, baris kosong memisahkan artikel."""
    with open(path, encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
    return [[s for s in b.split("\n") if s.strip()] for b in blocks if b.strip()]


class PretrainDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None, max_len: int = pretrain_cfg.max_len, use_cache: bool = True):
        self.max_len = max_len
        corpus = paths.pretrain_split[split]
        cache = paths.processed_dir / f"pretrain_{split}_blocks_{max_len}.pt"

        if use_cache and cache.exists() and cache.stat().st_mtime > max(
            corpus.stat().st_mtime, paths.tokenizer_file.stat().st_mtime
        ):
            data = torch.load(cache)
        else:
            data = self._build(corpus, tokenizer or load_tokenizer())
            if use_cache:
                torch.save(data, cache)
        self.tokens = data["tokens"]      # semua token disambung, int16 supaya hemat memori
        self.offsets = data["offsets"]    # blok ke-i = tokens[offsets[i]:offsets[i+1]]

    def _build(self, corpus, tokenizer) -> dict:
        body = self.max_len - 2           # sisakan tempat untuk [CLS] dan [SEP]
        docs = read_docs(corpus)
        flat = [s for d in docs for s in d]
        encoded = iter(tokenizer.encode_batch(flat, add_special_tokens=False))

        blocks = []
        for doc in docs:
            current = []
            for _ in doc:
                ids = next(encoded).ids
                # kalimat yang lebih panjang dari satu blok dipotong-potong
                for start in range(0, len(ids), body):
                    piece = ids[start:start + body]
                    if current and len(current) + len(piece) > body:
                        blocks.append(current)
                        current = []
                    current = current + piece
            if current:
                blocks.append(current)

        lengths = torch.tensor([len(b) for b in blocks], dtype=torch.long)
        offsets = torch.zeros(len(blocks) + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(lengths, 0)
        tokens = torch.tensor([i for b in blocks for i in b], dtype=torch.int16)
        return {"tokens": tokens, "offsets": offsets}

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        body = self.tokens[self.offsets[idx]:self.offsets[idx + 1]].tolist()
        input_ids, attention_mask = _pad([CLS_ID] + body + [SEP_ID], self.max_len)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TitleDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None, max_len: int = style_cfg.max_len, with_style: bool = True):
        tokenizer = tokenizer or load_tokenizer()
        df = pd.read_csv(paths.title_split[split])
        self.texts = df["text"].tolist()
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
        self.max_len = max_len
        self.with_style = with_style

        body = max_len - (3 if with_style else 2)
        encoded = tokenizer.encode_batch(self.texts, add_special_tokens=False)
        self.num_truncated = sum(len(e.ids) > body for e in encoded)

        ids, masks = [], []
        for e, label in zip(encoded, self.labels.tolist()):
            prefix = [CLS_ID, LABEL2STYLE_ID[label]] if with_style else [CLS_ID]
            i, m = _pad(prefix + e.ids[:body] + [SEP_ID], max_len)
            ids.append(i)
            masks.append(m)
        self.input_ids = torch.stack(ids)
        self.attention_mask = torch.stack(masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_pretrain_loader(split="train", batch_size=pretrain_cfg.batch_size, shuffle=None, num_workers=0, **kwargs):
    ds = PretrainDataset(split, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train") if shuffle is None else shuffle,
                      num_workers=num_workers, drop_last=False)


def build_title_loader(split="train", batch_size=style_cfg.batch_size, with_style=True, shuffle=None, num_workers=0, **kwargs):
    ds = TitleDataset(split, with_style=with_style, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train") if shuffle is None else shuffle,
                      num_workers=num_workers, drop_last=False)
