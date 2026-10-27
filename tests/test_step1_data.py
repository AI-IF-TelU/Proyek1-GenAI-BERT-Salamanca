"""
Cek kriteria selesai Step 1.

    python -m tests.test_step1_data       (tanpa pytest)
    pytest tests/test_step1_data.py -v    (kalau pytest terpasang)

Kriteria: satu batch DataLoader berbentuk (batch, 128) untuk pretraining dan
(batch, 48) untuk judul, ID token khusus benar, split tidak tumpang tindih dan
stratified, serta judul test tidak ada di korpus pretraining.
"""

import pandas as pd

from config import CB_ID, CLS_ID, NCB_ID, PAD_ID, SEP_ID, SPECIAL_TOKENS, paths, pretrain_cfg, style_cfg
from src.data.cleaning import clean_article, clean_title
from src.data.dataset import build_pretrain_loader, build_title_loader, read_docs
from src.data.tokenizer import load_tokenizer

BATCH = 8


def test_cleaning_examples():
    assert clean_title("Viral! Driver Ojol di Bekasi") == "viral! driver ojol di bekasi"
    sents = clean_article(
        "REPUBLIKA.CO.ID, JAKARTA -- Presiden ketiga Indonesia , di Jakarta wafat.,Ia dimakamkan hari ini."
        "Warga berduka cita. Baca juga: Judul Lain (iwd/iwd) tag tag"
    )
    assert sents[0] == "presiden ketiga indonesia, di jakarta wafat.", sents
    assert sents[1] == "ia dimakamkan hari ini.", sents
    assert not any("baca juga" in s or "iwd" in s for s in sents), sents


def test_special_token_ids():
    tok = load_tokenizer()
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i


def test_splits_disjoint_and_stratified():
    splits = {k: pd.read_csv(v) for k, v in paths.title_split.items()}
    texts = {k: set(df["text"]) for k, df in splits.items()}
    assert not texts["train"] & texts["val"]
    assert not texts["train"] & texts["test"]
    assert not texts["val"] & texts["test"]
    ratios = [df["label"].mean() for df in splits.values()]
    assert max(ratios) - min(ratios) < 0.01, ratios
    total = sum(len(df) for df in splits.values())
    assert abs(len(splits["train"]) / total - 0.8) < 0.01


def test_no_test_titles_in_pretrain_corpus():
    test_titles = set(pd.read_csv(paths.title_split["test"])["text"])
    for split in ("train", "val"):
        sentences = {s for d in read_docs(paths.pretrain_split[split]) for s in d}
        leaked = test_titles & sentences
        assert not leaked, f"{len(leaked)} judul test bocor ke {split}: {list(leaked)[:3]}"


def test_pretrain_batch_shape():
    batch = next(iter(build_pretrain_loader("val", batch_size=BATCH)))
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert ids.shape == (BATCH, pretrain_cfg.max_len), ids.shape
    assert mask.shape == ids.shape
    assert (ids[:, 0] == CLS_ID).all()
    assert ((ids == SEP_ID).sum(1) == 1).all()
    assert ((ids == PAD_ID) == (mask == 0)).all()


def test_title_batch_shape():
    batch = next(iter(build_title_loader("train", batch_size=BATCH, with_style=True, shuffle=False)))
    ids = batch["input_ids"]
    assert ids.shape == (BATCH, style_cfg.max_len), ids.shape
    assert batch["labels"].shape == (BATCH,)
    expected_style = [CB_ID if y == 1 else NCB_ID for y in batch["labels"].tolist()]
    assert ids[:, 1].tolist() == expected_style

    plain = next(iter(build_title_loader("train", batch_size=BATCH, with_style=False, shuffle=False)))
    assert plain["input_ids"].shape == (BATCH, style_cfg.max_len)
    assert not ((plain["input_ids"] == CB_ID) | (plain["input_ids"] == NCB_ID)).any()


def show_examples():
    tok = load_tokenizer()
    batch = next(iter(build_title_loader("train", batch_size=3, with_style=True, shuffle=False)))
    print("\nContoh batch judul (with_style=True):")
    for ids in batch["input_ids"]:
        print("  ", " ".join(tok.id_to_token(i) for i in ids.tolist() if i != PAD_ID))
    batch = next(iter(build_pretrain_loader("val", batch_size=1)))
    print("\nContoh blok pretraining:")
    print("  ", tok.decode(batch["input_ids"][0].tolist(), skip_special_tokens=False)[:400], "...")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    if not failed:
        show_examples()
    print(f"\n{len(tests) - failed}/{len(tests)} tes lulus")
    raise SystemExit(1 if failed else 0)
