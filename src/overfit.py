"""
Step 2 - Sanity check: model harus bisa overfit pada satu batch kecil sampai loss mendekati nol.

Jalankan dari root repo (tidak butuh dataset Kaggle, cukup data/splits dan tokenizer di repo):
    python -m src.overfit

Satu batch judul train (tanpa token gaya) di-mask sekali dengan seed tetap, lalu model dilatih
berulang-ulang pada batch yang sama dengan loss MLM + loss klasifikasi. Dropout dimatikan supaya
model bisa benar-benar menghafal. Kalau loss tidak turun ke ~0, berarti ada bug di arsitektur
(misalnya mask atensi salah, gradien tidak mengalir, atau weight tying rusak).

Output:
    reports/model_overfit.json         jumlah parameter, loss awal/akhir, akurasi akhir
    reports/figures/overfit_loss.png   grafik loss
"""

import json
import math
import time
from dataclasses import replace

import torch

from config import SEED, overfit_cfg, model_cfg, paths
from src.collator import mask_tokens
from src.data.dataset import build_title_loader
from src.model import IndoMiniBERT
from src.plots import plot_overfit


def run_overfit(steps=overfit_cfg.steps, batch_size=overfit_cfg.batch_size, lr=overfit_cfg.lr, log=True):
    torch.manual_seed(SEED)
    batch = next(iter(build_title_loader("train", batch_size=batch_size, with_style=False, shuffle=False)))
    input_ids, attention_mask, y = batch["input_ids"], batch["attention_mask"], batch["labels"]
    inputs, mlm_labels = mask_tokens(input_ids, generator=torch.Generator().manual_seed(SEED))

    model = IndoMiniBERT(replace(model_cfg, dropout=0.0, attention_dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"step": [], "mlm_loss": [], "cls_loss": []}

    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        mlm = model(inputs, attention_mask, labels=mlm_labels)
        cls = model(input_ids, attention_mask, labels=y, task="cls")
        loss = mlm["loss"] + cls["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["step"].append(step)
        history["mlm_loss"].append(mlm["loss"].item())
        history["cls_loss"].append(cls["loss"].item())
        if log and (step == 1 or step % 50 == 0):
            print(f"  step {step:4d}  mlm {mlm['loss'].item():.4f}  cls {cls['loss'].item():.4f}")

    model.eval()
    with torch.no_grad():
        mlm = model(inputs, attention_mask, labels=mlm_labels)
        cls = model(input_ids, attention_mask, task="cls")
    result = {
        "batch_shape": list(input_ids.shape),
        "masked_tokens": int((mlm_labels != -100).sum()),
        "steps": steps,
        "lr": lr,
        "parameters": model.num_parameters(),
        # loss awal MLM seharusnya ~ln(vocab) dan klasifikasi ~ln(2) karena prediksi awal hampir seragam
        "expected_initial_loss": {"mlm": round(math.log(model_cfg.vocab_size), 4), "cls": round(math.log(2), 4)},
        "initial_loss": {"mlm": round(history["mlm_loss"][0], 4), "cls": round(history["cls_loss"][0], 4)},
        "final_loss": {"mlm": round(history["mlm_loss"][-1], 6), "cls": round(history["cls_loss"][-1], 6)},
        "final_accuracy": {
            "mlm_masked_tokens": float((mlm["logits"].argmax(-1) == mlm["target"]).float().mean()),
            "cls": float((cls["logits"].argmax(-1) == y).float().mean()),
        },
        "target_loss": overfit_cfg.target_loss,
        "seconds": round(time.time() - start, 1),
    }
    result["passed"] = max(result["final_loss"].values()) < overfit_cfg.target_loss
    return result, history


def main():
    print(f"Tes overfit: {overfit_cfg.steps} step, batch {overfit_cfg.batch_size} judul, lr {overfit_cfg.lr}")
    result, history = run_overfit()
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out = paths.reports_dir / "model_overfit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    fig = plot_overfit(history, paths.reports_dir / "figures" / "overfit_loss.png")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nHasil disimpan di {out} dan {fig}")
    print("LULUS" if result["passed"] else "GAGAL: loss belum mendekati nol")


if __name__ == "__main__":
    main()
