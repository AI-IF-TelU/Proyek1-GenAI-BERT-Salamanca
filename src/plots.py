"""
Grafik untuk laporan (PNG): loss tes overfit (Step 2), loss pretraining dan perbandingan sweep (Step 3).

Semua fungsi menerima dict history yang sudah disimpan sebagai JSON, jadi grafik bisa dibuat ulang
tanpa training ulang:
    python -m src.plots reports/pretrain/<run>/history.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# palet kategorikal (urutan tetap) + tinta teks dan permukaan
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def _figure(ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.0), facecolor=SURFACE)
    return fig, axes


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_overfit(history: dict, out):
    """history: {"step": [...], "mlm_loss": [...], "cls_loss": [...]}"""
    fig, ax = _figure()
    for i, (key, name) in enumerate([("mlm_loss", "MLM"), ("cls_loss", "Klasifikasi")]):
        ax.plot(history["step"], history[key], color=SERIES[i], linewidth=2, label=name)
        v = history[key][-1]
        _end_label(ax, history["step"][-1], v, f"{name} {v:.4f}" if v >= 1e-3 else f"{name} {v:.1e}")
    ax.set_yscale("log")
    _style(ax, "Tes overfit satu batch", "step", "cross-entropy loss (skala log)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


def plot_pretrain(history: dict, out):
    """Kiri: train loss per log + val loss per epoch. Kanan: akurasi top-1 / top-5 token ter-mask (val)."""
    fig, (ax_loss, ax_acc) = _figure(2)
    train, val = history["train"], history["val"]

    if train:
        ax_loss.plot([r["step"] for r in train], [r["loss"] for r in train], color=SERIES[0], linewidth=2,
                     label="train")
    if val:
        xs, ys = [r["step"] for r in val], [r["loss"] for r in val]
        ax_loss.plot(xs, ys, color=SERIES[1], linewidth=2, marker="o", markersize=5, label="validasi")
        _end_label(ax_loss, xs[-1], ys[-1], f"{ys[-1]:.3f}")
    _style(ax_loss, "MLM loss", "step", "cross-entropy loss")
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=INK)

    if val:
        epochs = [r["epoch"] for r in val]
        for i, (key, name) in enumerate([("top1", "top-1"), ("top5", "top-5")]):
            ys = [r[key] * 100 for r in val]
            ax_acc.plot(epochs, ys, color=SERIES[i + 2], linewidth=2, marker="o", markersize=5, label=name)
            _end_label(ax_acc, epochs[-1], ys[-1], f"{name} {ys[-1]:.1f}%")
        ax_acc.set_xticks(epochs)
    _style(ax_acc, "Akurasi token ter-mask (validasi)", "epoch", "akurasi (%)")
    ax_acc.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


def plot_sweep(runs: dict, out):
    """runs: {nama_run: history}. Sumbu x = jumlah sampel dilihat, supaya batch size berbeda sebanding."""
    fig, ax = _figure()
    for i, (name, history) in enumerate(runs.items()):
        bs = history["config"]["batch_size"]
        xs = [r["step"] * bs for r in history["train"]]
        ys = [r["loss"] for r in history["train"]]
        if not xs:
            continue
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, label=name)
    _style(ax, "Sweep learning rate x batch size (train loss)", "jumlah sampel dilihat", "cross-entropy loss")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        path = Path(arg)
        with open(path, encoding="utf-8") as f:
            print(plot_pretrain(json.load(f), path.with_name("loss.png")))
