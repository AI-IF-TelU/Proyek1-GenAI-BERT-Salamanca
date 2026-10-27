"""
Step 3 - Eksperimen 3 variasi learning rate x 2 variasi batch size (config.PretrainConfig.sweep_*).

    python -m src.sweep                    # setiap kombinasi dilatih sweep_epochs epoch
    python -m src.sweep --max-steps 300    # lebih cepat: batasi step per epoch

Setiap kombinasi disimpan sebagai run "sweep_lr<lr>_bs<bs>" (checkpoint dan history seperti src.pretrain).
Ringkasan:
    reports/pretrain/sweep_summary.json, reports/pretrain/sweep_summary.md
    reports/figures/pretrain_sweep.png
Setelah itu latih konfigurasi terbaik secara penuh, misalnya:
    python -m src.pretrain --lr <lr_terbaik> --batch-size <bs_terbaik> --run-name main
"""

import argparse

from config import paths, pretrain_cfg
from src.plots import plot_sweep
from src.pretrain import save_json, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=pretrain_cfg.sweep_epochs)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=pretrain_cfg.num_workers)
    a = parser.parse_args()

    runs, rows = {}, []
    for bs in pretrain_cfg.sweep_batch_sizes:
        for lr in pretrain_cfg.sweep_lrs:
            name = f"sweep_lr{lr:g}_bs{bs}"
            history = train(lr=lr, batch_size=bs, epochs=a.epochs, run_name=name, max_steps=a.max_steps,
                            max_eval_batches=a.max_eval_batches, num_workers=a.num_workers)
            runs[name] = history
            rows.append({"run": name, "lr": lr, "batch_size": bs, "samples_seen": history["config"]["total_steps"] * bs,
                         **{k: history["val"][-1][k] for k in ("loss", "ppl", "top1", "top5", "minutes")}})

    rows.sort(key=lambda r: r["loss"])
    out_dir = paths.reports_dir / "pretrain"
    save_json({"best": rows[0], "runs": rows}, out_dir / "sweep_summary.json")
    lines = ["| run | lr | batch | val loss | ppl | top-1 | top-5 | menit |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['lr']:g} | {r['batch_size']} | {r['loss']} | {r['ppl']} | "
              f"{r['top1'] * 100:.1f}% | {r['top5'] * 100:.1f}% | {r['minutes']} |" for r in rows]
    (out_dir / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_sweep(runs, paths.reports_dir / "figures" / "pretrain_sweep.png")

    print("\n".join(lines))
    best = rows[0]
    print(f"\nTerbaik: lr={best['lr']:g}, batch={best['batch_size']}. Lanjutkan training penuh:")
    print(f"    python -m src.pretrain --lr {best['lr']:g} --batch-size {best['batch_size']} --run-name main")


if __name__ == "__main__":
    main()
