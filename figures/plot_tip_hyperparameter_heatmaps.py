#!/usr/bin/env python3
"""Plot TIP validation-MSE heatmaps from the hyperparameter grid results.

Run from anywhere inside the project environment:

    uv run python figures/plot_tip_hyperparameter_heatmaps.py

By default, a PNG file is written next to this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "results" / "tip_grid_Supply_Chain_Customer_D_SAITS_Toto2.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tip_hyperparameter_heatmaps"
DEFAULT_TRAIN_INPUT = (
    ROOT / "results" / "tip_grid_Supply_Chain_Customer_D_SAITS_Toto2_train.json"
)
DEFAULT_TRAIN_OUTPUT = (
    Path(__file__).resolve().parent / "tip_train_prediction_heatmaps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", choices=("validation", "training"), default="validation",
        help="Prediction MSE to plot (default: validation)",
    )
    parser.add_argument(
        "--input", type=Path,
        help="TIP grid JSON file; defaults to the selected metric's result file",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path without an extension; a PNG file is generated",
    )
    return parser.parse_args()


def valid_mse(row: dict, metric_key: str) -> float | None:
    value = row.get(metric_key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def epsilon_label(value: float) -> str:
    exponent = round(math.log10(value))
    if math.isclose(value, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"
    return f"{value:g}"


def load_grid(
    path: Path, metric_key: str,
) -> tuple[list[dict], list[int], list[float], list[float]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")

    required = {"k", "epsilon", "confidence_delta", metric_key}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")

    k_values = sorted({int(row["k"]) for row in rows})
    epsilon_values = sorted({float(row["epsilon"]) for row in rows})
    delta_values = sorted({float(row["confidence_delta"]) for row in rows})
    if len(k_values) != 3:
        raise ValueError(f"Expected three k values, found {k_values}")
    return rows, k_values, epsilon_values, delta_values


def main() -> None:
    args = parse_args()
    is_training = args.metric == "training"
    metric_key = "train_pred_mse" if is_training else "val_pred_mse"
    metric_label = "Training Prediction MSE" if is_training else "Validation Prediction MSE"
    default_input = DEFAULT_TRAIN_INPUT if is_training else DEFAULT_INPUT
    default_output = DEFAULT_TRAIN_OUTPUT if is_training else DEFAULT_OUTPUT
    input_path = (args.input or default_input).expanduser().resolve()
    rows, k_values, epsilon_values, delta_values = load_grid(
        input_path, metric_key
    )

    lookup: dict[tuple[int, float, float], float] = {}
    for row in rows:
        key = (
            int(row["k"]),
            float(row["epsilon"]),
            float(row["confidence_delta"]),
        )
        value = valid_mse(row, metric_key)
        if value is not None:
            if key in lookup:
                raise ValueError(f"Duplicate completed grid point: {key}")
            lookup[key] = value
    if not lookup:
        raise ValueError("No finite validation prediction MSE values were found")

    best_key = min(lookup, key=lookup.get)
    finite_values = np.asarray(list(lookup.values()), dtype=float)
    color_min = float(finite_values.min())
    color_max = float(finite_values.max())
    if math.isclose(color_min, color_max):
        color_max = color_min + 1e-6

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.35), sharex=True, sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = None

    for panel_index, (axis, k_value) in enumerate(zip(axes, k_values, strict=True)):
        matrix = np.full((len(delta_values), len(epsilon_values)), np.nan)
        for row_index, delta in enumerate(delta_values):
            for column_index, epsilon in enumerate(epsilon_values):
                matrix[row_index, column_index] = lookup.get(
                    (k_value, epsilon, delta), np.nan
                )

        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=color_min,
            vmax=color_max,
            interpolation="nearest",
        )
        axis.set_title(rf"$k = {k_value}$")
        axis.set_xlabel(r"Perturbation scale $\epsilon$")
        if panel_index == 0:
            axis.set_ylabel(r"Confidence level $\delta$")
        axis.set_xticks(range(len(epsilon_values)))
        axis.set_xticklabels([epsilon_label(value) for value in epsilon_values])
        axis.set_yticks(range(len(delta_values)))
        axis.set_yticklabels([f"{value:g}" for value in delta_values])
        axis.set_xticks(np.arange(-0.5, len(epsilon_values), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(delta_values), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)

        midpoint = (color_min + color_max) / 2.0
        for row_index, delta in enumerate(delta_values):
            for column_index, epsilon in enumerate(epsilon_values):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    text_color = "white" if value < midpoint else "black"
                    axis.text(
                        column_index, row_index, f"{value:.3f}",
                        ha="center", va="center", color=text_color,
                        fontsize=8.5, fontweight="medium",
                    )
                else:
                    axis.text(
                        column_index, row_index, "N/A",
                        ha="center", va="center", color="#555555", fontsize=8.5,
                    )

        if best_key[0] == k_value:
            best_column = epsilon_values.index(best_key[1])
            best_row = delta_values.index(best_key[2])
            axis.add_patch(Rectangle(
                (best_column - 0.47, best_row - 0.47), 0.94, 0.94,
                fill=False, edgecolor="#e63946", linewidth=2.4,
            ))

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label(metric_label)

    output_stem = (args.output or default_output).expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(lookup)} completed grid points from: {input_path}")
    print(
        f"{args.metric.capitalize()}-best configuration: "
        f"k={best_key[0]}, epsilon={best_key[1]:g}, "
        f"delta={best_key[2]:g}, MSE={lookup[best_key]:.6f}"
    )
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
