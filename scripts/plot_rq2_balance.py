#!/usr/bin/env python3
"""绘制 RQ2 的 B2/B1 插补--预测平衡散点图。

每个点对应一个（数据集，插补模型，预测模型）配置：
  x = B2 测试集插补 MSE / B1 测试集插补 MSE
  y = B2 测试集预测 MSE / B1 测试集预测 MSE

自然缺失数据集或缺少成对结果的配置不进入二维散点图，但会写入明细 CSV，
并在终端报告排除原因。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tip-matplotlib-cache")
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fill_table6 import (  # noqa: E402
    DATASETS,
    IMPUTERS,
    NATURAL_MISSING_DATASETS,
    PREDICTORS,
    collect_logs,
)


PREDICTOR_COLORS = {
    "Sundial": "#0072B2",
    "Toto2": "#D55E00",
    "Chronos-2": "#009E73",
}
IMPUTER_MARKERS = {
    "BRITS": "o",
    "SAITS": "s",
    "DeepMVI": "^",
    "ModernTCN": "D",
    "TimesNet": "P",
    "TSLANet": "X",
}

PLOT_EXCLUSIONS = {
    ("EWELD Load", "SAITS", "Sundial"): "upper_right_outlier",
}


def classify_quadrant(x: float, y: float) -> str:
    if math.isclose(x, 1.0) or math.isclose(y, 1.0):
        return "boundary"
    if x < 1.0 and y < 1.0:
        return "both_improve"
    if x > 1.0 and y < 1.0:
        return "prediction_tradeoff"
    if x < 1.0 and y > 1.0:
        return "imputation_mismatch"
    return "both_degrade"


def collect_pairs(results: dict) -> tuple[list[dict], list[dict]]:
    points: list[dict] = []
    excluded: list[dict] = []
    for dataset in DATASETS:
        for imputer in IMPUTERS:
            for predictor in PREDICTORS:
                base = {
                    "dataset": dataset,
                    "imputer": imputer,
                    "predictor": predictor,
                }
                exclusion_reason = PLOT_EXCLUSIONS.get((dataset, imputer, predictor))
                if exclusion_reason is not None:
                    excluded.append({**base, "reason": exclusion_reason})
                    continue
                b1 = results.get(("B1", dataset, imputer, predictor))
                b2 = results.get(("B2", dataset, imputer, predictor))
                if dataset in NATURAL_MISSING_DATASETS:
                    excluded.append({**base, "reason": "natural_missing_no_imp_mse"})
                    continue
                if b1 is None or b2 is None:
                    excluded.append({**base, "reason": "missing_b1_or_b2_log"})
                    continue
                values = {
                    "b1_imp_mse": b1.get("imp_mse"),
                    "b2_imp_mse": b2.get("imp_mse"),
                    "b1_pred_mse": b1.get("pred_mse"),
                    "b2_pred_mse": b2.get("pred_mse"),
                }
                if any(value is None for value in values.values()):
                    excluded.append({**base, **values, "reason": "missing_test_mse"})
                    continue
                if values["b1_imp_mse"] <= 0 or values["b1_pred_mse"] <= 0:
                    excluded.append({**base, **values, "reason": "nonpositive_b1_mse"})
                    continue
                imp_ratio = values["b2_imp_mse"] / values["b1_imp_mse"]
                pred_ratio = values["b2_pred_mse"] / values["b1_pred_mse"]
                points.append({
                    **base,
                    **values,
                    "imp_ratio": imp_ratio,
                    "pred_ratio": pred_ratio,
                    "quadrant": classify_quadrant(imp_ratio, pred_ratio),
                })
    return points, excluded


def write_csv(path: Path, points: list[dict], excluded: list[dict]) -> None:
    fields = [
        "included", "dataset", "imputer", "predictor",
        "b1_imp_mse", "b2_imp_mse", "b1_pred_mse", "b2_pred_mse",
        "imp_ratio", "pred_ratio", "quadrant", "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({"included": 1, **point})
        for item in excluded:
            writer.writerow({"included": 0, **item})


def padded_limits(values: list[float]) -> tuple[float, float]:
    low = min(values + [1.0])
    high = max(values + [1.0])
    span = max(high - low, 0.05)
    return max(0.0, low - 0.10 * span), high + 0.10 * span


def plot_balance(points: list[dict], output_prefix: Path, dpi: int) -> None:
    if not points:
        raise RuntimeError("没有可用于 RQ2 二维平衡图的完整 B1/B2 配对")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)

    for point in points:
        ax.scatter(
            point["imp_ratio"],
            point["pred_ratio"],
            s=62,
            marker=IMPUTER_MARKERS[point["imputer"]],
            facecolor=PREDICTOR_COLORS[point["predictor"]],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.90,
            zorder=3,
        )

    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlim(*padded_limits([point["imp_ratio"] for point in points]))
    ax.set_ylim(*padded_limits([point["pred_ratio"] for point in points]))
    ax.set_xlabel(r"Relative imputation MSE  ($\mathrm{B2}/\mathrm{B1}$)")
    ax.set_ylabel(r"Relative prediction MSE  ($\mathrm{B2}/\mathrm{B1}$)")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, color="0.88", linewidth=0.6, zorder=0)

    color_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=color, markeredgecolor="white", label=name)
        for name, color in PREDICTOR_COLORS.items()
    ]
    shape_handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=7,
               markerfacecolor="0.45", markeredgecolor="white", label=name)
        for name, marker in IMPUTER_MARKERS.items()
    ]
    predictor_legend = ax.legend(
        handles=color_handles, title="Predictor (color)",
        loc="upper left", bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0.0, frameon=True, framealpha=0.88,
        fontsize=11, title_fontsize=11,
    )
    ax.add_artist(predictor_legend)
    ax.legend(
        handles=shape_handles, title="Imputer (marker)",
        loc="upper right", bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.0, frameon=True, framealpha=0.88,
        fontsize=11, title_fontsize=11, ncol=2,
    )

    counts = Counter(point["quadrant"] for point in points)
    quadrant_total = len(points) - counts["boundary"]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    quadrant_positions = {
        "both_improve": ((xmin + 1.0) / 2, (ymin + 1.0) / 2),
        "prediction_tradeoff": ((1.0 + xmax) / 2, (ymin + 1.0) / 2),
        "imputation_mismatch": ((xmin + 1.0) / 2, 1.0 + 0.22 * (ymax - 1.0)),
        "both_degrade": ((1.0 + xmax) / 2, 1.0 + 0.22 * (ymax - 1.0)),
    }
    for quadrant, (xpos, ypos) in quadrant_positions.items():
        ax.text(
            xpos, ypos, f"{counts[quadrant] / quadrant_total:.1%}",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="0.28", alpha=0.72, zorder=2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.58},
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument(
        "--output-prefix", type=Path,
        default=ROOT / "paper" / "figures" / "rq2_balance",
        help="输出路径（不含扩展名），同时生成 PDF、PNG 和 CSV",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = collect_logs(args.logs_dir, args.verbose)
    points, excluded = collect_pairs(results)
    plot_balance(points, args.output_prefix, args.dpi)
    csv_path = args.output_prefix.with_suffix(".csv")
    write_csv(csv_path, points, excluded)

    reasons = Counter(item["reason"] for item in excluded)
    print(f"Included configurations: {len(points)}")
    print(f"Excluded configurations: {len(excluded)}")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    print(f"PDF: {args.output_prefix.with_suffix('.pdf')}")
    print(f"PNG: {args.output_prefix.with_suffix('.png')}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
