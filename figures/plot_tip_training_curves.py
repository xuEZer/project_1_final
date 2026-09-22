"""Plot per-epoch training curves (imputation/prediction, train/val) by stage.

Reads a snapshot of the ablation batch metrics (SQLite) plus a run-id mapping
JSON produced alongside it, and draws the mean-over-seeds curves for each
TIP branch.  Usage:

    python figures/plot_tip_training_curves.py \
        --db tmp/ablation_metrics.sqlite3 \
        --mapping tmp/ablation_mapping.json \
        --cell azure2019_I_5T SAITS Toto2 \
        --out figures/tip_training_curves.png
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {
    "tr_imp": '["loss", "train_imp_loss"]',
    "val_imp": '["loss", "val_imp_loss"]',
    "tr_pred": '["loss", "train_pred_loss"]',
    "val_pred": '["loss", "val_pred_loss"]',
}
TITLES = {
    "tr_imp": "Train imputation MSE",
    "val_imp": "Validation imputation MSE",
    "tr_pred": "Train forecast MSE",
    "val_pred": "Validation forecast MSE",
}
STAGES = ["B1", "B2", "B3", "B4", "B5"]
STYLES = {
    "B1": {"color": "#666666", "ls": "-", "lw": 2.0},
    "B2": {"color": "#1f77b4", "ls": "-", "lw": 2.2},
    "B3": {"color": "#9467bd", "ls": "--", "lw": 1.6},
    "B4": {"color": "#2ca02c", "ls": "--", "lw": 1.6},
    "B5": {"color": "#d62728", "ls": "-.", "lw": 1.6},
}


def load_curves(db: Path, mapping: dict, cell: tuple[str, str, str]) -> dict:
    con = sqlite3.connect(db)
    curves = defaultdict(lambda: defaultdict(dict))  # stage -> seed -> metric
    for rid, meta in mapping.items():
        if tuple(meta["cell"]) != cell or meta["stage"] not in STAGES:
            continue
        for name, mp in METRICS.items():
            rows = con.execute(
                "SELECT step, value FROM metrics WHERE run_id=? AND "
                "stage_path='[1]' AND metric_path=? ORDER BY step",
                (rid, mp),
            ).fetchall()
            if rows:
                curves[meta["stage"]][meta["seed"]][name] = [v for _, v in rows]
    con.close()
    return curves


def mean_curve(seed_curves: dict, metric: str) -> tuple[list[int], list[float], int]:
    cols = [
        s[metric]
        for s in seed_curves.values()
        if metric in s and len(s[metric]) == max(
            (len(x.get(metric, [])) for x in seed_curves.values()), default=0
        )
    ]
    if not cols:
        return [], [], 0
    n = len(cols[0])
    return list(range(1, n + 1)), [st.mean(c[i] for c in cols) for i in range(n)], len(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--cell", nargs=3, metavar=("DATASET", "IMPUTER", "PREDICTOR"),
                        default=["azure2019_I_5T", "SAITS", "Toto2"])
    parser.add_argument("--out", type=Path, default=Path("figures/tip_training_curves.png"))
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    curves = load_curves(args.db, mapping, tuple(args.cell))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric in zip(axes.flat, METRICS):
        for stage in STAGES:
            seeds = curves.get(stage, {})
            if not seeds:
                continue
            epochs, means, n = mean_curve(seeds, metric)
            if not epochs:
                continue
            style = STYLES[stage]
            ax.plot(epochs, means, label=f"{stage} (n={n})", **style)
        ax.set_title(TITLES[metric])
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("TIP ablation training curves — " + "/".join(args.cell))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
