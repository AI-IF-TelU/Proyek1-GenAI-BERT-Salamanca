"""
Cek kriteria Step 3: rasio dynamic masking dan token khusus.

    python -m tests.test_step3_masking
    pytest tests/test_step3_masking.py -v

Dengan ~200 ribu token, rasio empiris seharusnya sangat dekat dengan 15% / 80% / 10% / 10%.
"""

import torch

from config import CB_ID, CLS_ID, MASK_ID, NCB_ID, PAD_ID, SEP_ID, model_cfg, pretrain_cfg
from src.collator import MLMCollator, mask_tokens
from src.model import IGNORE_INDEX

TOL = 0.01


def _batch(n=1600, length=128):
    torch.manual_seed(0)
    ids = torch.randint(NCB_ID + 1, model_cfg.vocab_size, (n, length))
    sep = length * 3 // 4                 # sisa 1/4 terakhir = padding
    ids[:, 0], ids[:, 1], ids[:, sep] = CLS_ID, CB_ID, SEP_ID
    ids[: n // 2, 2] = NCB_ID
    ids[:, sep + 1:] = PAD_ID
    return ids


def test_mask_ratios():
    ids = _batch()
    inputs, labels = mask_tokens(ids)
    maskable = ids > NCB_ID
    selected = labels != IGNORE_INDEX

    p = selected.sum() / maskable.sum()
    assert abs(p - pretrain_cfg.mask_prob) < TOL, f"rasio terpilih {p:.4f}"

    n = selected.sum()
    as_mask = ((inputs == MASK_ID) & selected).sum() / n
    kept = ((inputs == ids) & selected).sum() / n
    as_random = ((inputs != MASK_ID) & (inputs != ids) & selected).sum() / n
    print(f"  terpilih {p:.4f} | [MASK] {as_mask:.4f} | acak {as_random:.4f} | tetap {kept:.4f}")
    assert abs(as_mask - pretrain_cfg.mask_replace_prob) < TOL
    assert abs(as_random - pretrain_cfg.random_replace_prob) < TOL
    assert abs(kept - (1 - pretrain_cfg.mask_replace_prob - pretrain_cfg.random_replace_prob)) < TOL


def test_special_tokens_never_masked():
    ids = _batch()
    inputs, labels = mask_tokens(ids)
    special = ids <= NCB_ID
    assert (labels[special] == IGNORE_INDEX).all()
    assert torch.equal(inputs[special], ids[special])
    # token acak tidak boleh berupa token khusus
    assert (inputs[~special] > NCB_ID).logical_or(inputs[~special] == MASK_ID).all()


def test_labels_hold_original_tokens():
    ids = _batch()
    inputs, labels = mask_tokens(ids)
    selected = labels != IGNORE_INDEX
    assert torch.equal(labels[selected], ids[selected])
    assert torch.equal(inputs[~selected], ids[~selected])
    assert not torch.equal(inputs, ids)  # input asli tidak diubah in-place, hasilnya tensor baru


def test_dynamic_and_seeded():
    ids = _batch(n=64)
    a, _ = mask_tokens(ids)
    b, _ = mask_tokens(ids)
    assert not torch.equal(a, b), "masking harus berbeda tiap pemanggilan (dynamic)"
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.equal(mask_tokens(ids, generator=g1)[0], mask_tokens(ids, generator=g2)[0])


def test_collator():
    examples = [{"input_ids": row, "attention_mask": (row != PAD_ID).long(), "labels": torch.tensor(1)}
                for row in _batch(n=8, length=48)]
    batch = MLMCollator()(examples)
    assert batch["input_ids"].shape == batch["labels"].shape == (8, 48)
    assert batch["style_labels"].tolist() == [1] * 8
    assert (batch["input_ids"][:, 1] == CB_ID).all(), "token gaya tidak boleh di-mask"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tes lulus")
    raise SystemExit(1 if failed else 0)
