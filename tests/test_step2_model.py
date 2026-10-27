"""
Cek kriteria selesai Step 2.

    python -m tests.test_step2_model       (tanpa pytest)
    pytest tests/test_step2_model.py -v    (kalau pytest terpasang)

Kriteria: forward dan backward berjalan, bentuk output benar, weight tying aktif, padding
benar-benar diabaikan atensi, jumlah parameter tercatat, dan tes overfit satu batch lulus.
"""

from dataclasses import replace

import torch

from config import CLS_ID, PAD_ID, SEP_ID, model_cfg, overfit_cfg, pretrain_cfg, style_cfg
from src.model import IGNORE_INDEX, IndoMiniBERT, load_model, save_model
from src.overfit import run_overfit

B = 4


def _batch(length, n_pad=0):
    torch.manual_seed(0)
    ids = torch.randint(10, model_cfg.vocab_size, (B, length))
    ids[:, 0], ids[:, length - n_pad - 1] = CLS_ID, SEP_ID
    mask = torch.ones_like(ids)
    if n_pad:
        ids[:, -n_pad:] = PAD_ID
        mask[:, -n_pad:] = 0
    return ids, mask


def test_forward_shapes():
    model = IndoMiniBERT()
    for length in (pretrain_cfg.max_len, style_cfg.max_len):
        ids, mask = _batch(length, n_pad=5)
        out = model(ids, mask, output_attentions=True)
        assert out["hidden_states"].shape == (B, length, model_cfg.hidden_size)
        assert out["logits"].shape == (B, length, model_cfg.vocab_size)
        assert len(out["attentions"]) == model_cfg.num_layers
        assert out["attentions"][0].shape == (B, model_cfg.num_heads, length, length)
        assert model(ids, mask, task="cls")["logits"].shape == (B, model_cfg.num_labels)


def test_mlm_loss_only_on_masked_positions():
    model = IndoMiniBERT()
    ids, mask = _batch(style_cfg.max_len)
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:, 3] = ids[:, 3]
    out = model(ids, mask, labels=labels)
    assert out["logits"].shape == (B, model_cfg.vocab_size)
    assert torch.equal(out["target"], ids[:, 3])
    empty = model(ids, mask, labels=torch.full_like(ids, IGNORE_INDEX))["loss"]
    assert empty.item() == 0.0 and not torch.isnan(empty)


def test_backward_reaches_all_parameters():
    model = IndoMiniBERT()
    ids, mask = _batch(style_cfg.max_len, n_pad=5)
    labels = ids.masked_fill(torch.rand(ids.shape) > 0.3, IGNORE_INDEX)
    loss = model(ids, mask, labels=labels)["loss"] + model(ids, mask, task="cls", labels=torch.tensor([0, 1, 0, 1]))["loss"]
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing


def test_weight_tying():
    model = IndoMiniBERT()
    assert model.mlm_head.decoder.weight is model.bert.embeddings.token.weight
    unique = {id(p) for p in model.parameters()}
    assert len(unique) == len(list(model.parameters()))


def test_padding_is_ignored():
    # isi token [PAD] diganti token sembarang; output di posisi non-pad harus identik
    model = IndoMiniBERT().eval()
    ids, mask = _batch(style_cfg.max_len, n_pad=10)
    noisy = ids.clone()
    noisy[:, -10:] = torch.randint(10, model_cfg.vocab_size, (B, 10))
    with torch.no_grad():
        a = model(ids, mask, output_attentions=True)
        b = model(noisy, mask)
    real = mask.bool()
    assert torch.allclose(a["hidden_states"][real], b["hidden_states"][real], atol=1e-5)
    probs = a["attentions"][0]
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)
    assert probs[..., -10:].abs().max() < 1e-6


def test_bidirectional_attention():
    # BERT dua arah: mengubah token di akhir kalimat harus mengubah representasi token pertama
    model = IndoMiniBERT().eval()
    ids, mask = _batch(style_cfg.max_len)
    changed = ids.clone()
    changed[:, -2] = (changed[:, -2] + 1) % model_cfg.vocab_size
    with torch.no_grad():
        a = model(ids, mask)["hidden_states"][:, 1]
        b = model(changed, mask)["hidden_states"][:, 1]
    assert not torch.allclose(a, b)


def test_parameter_count():
    counts = IndoMiniBERT().num_parameters()
    print("  jumlah parameter:", counts)
    assert 5_000_000 < counts["total"] < 10_000_000, counts
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")


def test_save_and_load():
    import tempfile
    from pathlib import Path

    model = IndoMiniBERT(replace(model_cfg, num_layers=2)).eval()
    ids, mask = _batch(style_cfg.max_len)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        save_model(model, path)
        loaded = load_model(path).eval()
    assert loaded.cfg.num_layers == 2
    with torch.no_grad():
        assert torch.allclose(model(ids, mask)["logits"], loaded(ids, mask)["logits"])


def test_overfit_one_batch():
    result, _ = run_overfit(log=False)
    print(f"  loss akhir: {result['final_loss']}  ({result['seconds']} detik)")
    assert result["passed"], result["final_loss"]
    assert max(result["final_loss"].values()) < overfit_cfg.target_loss


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
