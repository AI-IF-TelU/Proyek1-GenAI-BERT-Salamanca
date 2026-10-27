"""
Unit & Integration Tests untuk Step 4: MLM Bergaya dan Classifier Clickbait.

Memvalidasi kriteria selesai dari Proposal Step 4:
  1. Generator bergaya (style_mlm) tidak pernah me-mask token gaya (posisi 1: [CB] / [NCB])
  2. Model classifier (BertForSequenceClassification) memproses judul tanpa token gaya
     dan menghasilkan logits 2-kelas (clickbait vs non-clickbait)
  3. Checkpoint Step 4 dapat dimuat dengan tepat
  4. Laporan ablation dan evaluasi Step 4 berisi metrik yang disyaratkan (Akurasi, Macro F1, Confusion Matrix)
"""

import json
from pathlib import Path
import pytest
import torch

from config import CB_ID, CLS_ID, NCB_ID, PAD_ID, SEP_ID, model_cfg, paths
from src.collator import DynamicMaskingCollator
from src.model import BertForMaskedLM, BertForSequenceClassification


def test_style_tokens_never_masked():
    """Validasi bahwa token gaya [CB] / [NCB] di posisi 1 dan [CLS] di posisi 0 tidak pernah di-mask."""
    collator = DynamicMaskingCollator(protected_positions=(0, 1))

    # Simulasi 100 sampel berlabel dengan token gaya di posisi 1
    seq_len = 48
    input_ids_cb = torch.full((50, seq_len), 100, dtype=torch.long)
    input_ids_cb[:, 0] = CLS_ID
    input_ids_cb[:, 1] = CB_ID
    input_ids_cb[:, 20] = SEP_ID
    input_ids_cb[:, 21:] = PAD_ID

    input_ids_ncb = torch.full((50, seq_len), 100, dtype=torch.long)
    input_ids_ncb[:, 0] = CLS_ID
    input_ids_ncb[:, 1] = NCB_ID
    input_ids_ncb[:, 20] = SEP_ID
    input_ids_ncb[:, 21:] = PAD_ID

    all_ids = torch.cat([input_ids_cb, input_ids_ncb], dim=0)
    attn_mask = (all_ids != PAD_ID).long()

    batch = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(all_ids, attn_mask)
    ]

    masked_batch = collator(batch)
    labels = masked_batch["labels"]  # -100 jika tidak di-mask

    # Posisi 0 ([CLS]) dan posisi 1 ([CB]/[NCB]) TIDAK boleh di-mask (label harus -100)
    assert (labels[:, 0] == -100).all(), "[CLS] pada posisi 0 tidak sengaja di-mask!"
    assert (labels[:, 1] == -100).all(), "Token gaya ([CB]/[NCB]) pada posisi 1 tidak sengaja di-mask!"

    # Token asli di posisi 0 dan 1 pada input_ids juga tidak boleh diubah
    assert (masked_batch["input_ids"][:, 0] == CLS_ID).all()
    assert (masked_batch["input_ids"][:50, 1] == CB_ID).all()
    assert (masked_batch["input_ids"][50:, 1] == NCB_ID).all()


def test_classifier_architecture_forward_backward():
    """Validasi arsitektur classifier clickbait dan output 2 kelas."""
    model = BertForSequenceClassification(num_labels=2)
    batch_size = 4
    seq_len = 32
    dummy_input = torch.randint(low=7, high=1000, size=(batch_size, seq_len))
    dummy_input[:, 0] = CLS_ID
    dummy_mask = torch.ones_like(dummy_input)

    logits = model(dummy_input, dummy_mask)
    assert logits.shape == (batch_size, 2), f"Logits shape harus (batch, 2), didapat {logits.shape}"

    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # Pastikan gradient mengalir ke bobot classifier dan encoder
    assert model.classifier.weight.grad is not None
    assert model.bert.embeddings.token_embedding.weight.grad is not None


def test_style_mlm_checkpoint_loading():
    """Validasi bahwa checkpoint style_mlm dapat dimuat tanpa error."""
    ckpt_path = paths.checkpoints_dir / "style_mlm.pt"
    if not ckpt_path.exists():
        pytest.skip("Checkpoint style_mlm.pt belum dibuat")

    model = BertForMaskedLM()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state" in ckpt
    model.load_state_dict(ckpt["model_state"], strict=True)
    assert model.mlm_head.weight.shape == (model_cfg.vocab_size, model_cfg.hidden_size)


def test_classifier_checkpoints_loading():
    """Validasi bahwa kedua checkpoint classifier (random init dan pretrained) dapat dimuat."""
    rand_path = paths.checkpoints_dir / "classifier_random_init.pt"
    pre_path = paths.checkpoints_dir / "classifier_pretrained.pt"

    if not rand_path.exists() or not pre_path.exists():
        pytest.skip("Checkpoint classifier belum lengkap")

    rand_model = BertForSequenceClassification(num_labels=2)
    rand_ckpt = torch.load(rand_path, map_location="cpu")
    rand_model.load_state_dict(rand_ckpt["model_state"], strict=True)

    pre_model = BertForSequenceClassification(num_labels=2)
    pre_ckpt = torch.load(pre_path, map_location="cpu")
    pre_model.load_state_dict(pre_ckpt["model_state"], strict=True)


def test_ablation_report_contents():
    """Validasi kriteria selesai Step 4 dari proposal: laporan perbandingan F1 pretrained vs random."""
    report_path = paths.reports_dir / "style_classifier_ablation.json"
    if not report_path.exists():
        pytest.skip("Laporan style_classifier_ablation.json belum dibuat")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "random_init" in data
    assert "pretrained" in data
    assert "comparison_table" in data

    comp = data["comparison_table"]
    variants = [row["variant"] for row in comp]
    assert "random_init" in variants
    assert "pretrained" in variants

    for row in comp:
        assert "test_accuracy" in row
        assert "test_f1_macro" in row
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert 0.0 <= row["test_f1_macro"] <= 1.0
        assert "test_confusion_matrix" in row
