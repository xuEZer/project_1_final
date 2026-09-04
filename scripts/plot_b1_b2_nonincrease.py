"""Plot paired B1 and B2 prediction non-increase rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fill_table6 import collect_logs


def collect_pairs(logs_dir: Path) -> list[dict[str, str | float]]:
    results = collect_logs(logs_dir, verbose=False)
    pairs: list[dict[str, str | float]] = []
    for key, b1 in results.items():
        stage, dataset, imputer, predictor = key
        if stage != "B1":
            continue
        b2 = results.get(("B2", dataset, imputer, predictor))
        if b2 is None:
            continue
        b1_rate = b1.get("nonincrease_rate")
        b2_rate = b2.get("nonincrease_rate")
        if b1_rate is None or b2_rate is None:
            continue
        pairs.append(
            {
                "dataset": dataset,
                "imputer": imputer,
                "predictor": predictor,
                "b1_nonincrease_rate": float(b1_rate),
                "b2_nonincrease_rate": float(b2_rate),
            }
        )
    return sorted(
        pairs,
        key=lambda row: (str(row["dataset"]), str(row["imputer"]), str(row["predictor"])),
    )


def write_pairs(path: Path, pairs: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)


def draw_figure(pairs: list[dict[str, str | float]], output: Path) -> None:
    x = np.asarray([float(row["b1_nonincrease_rate"]) for row in pairs])
    y = np.asarray([float(row["b2_nonincrease_rate"]) for row in pairs])
    boundary = np.linspace(0.0, 100.0, 401)

    figure, axis = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    axis.fill_between(
        boundary,
        boundary,
        100.0,
        color="#DCFCE7",
        alpha=0.72,
        linewidth=0,
        zorder=0,
    )
    axis.fill_between(
        boundary,
        0.0,
        boundary,
        color="#FEE2E2",
        alpha=0.72,
        linewidth=0,
        zorder=0,
    )
    axis.plot(
        boundary,
        boundary,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=r"$y=x$",
        zorder=2,
    )
    axis.scatter(
        x,
        y,
        s=46,
        color="#2563EB",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.86,
        zorder=3,
    )
    axis.set_xlim(0.0, 102.0)
    axis.set_ylim(0.0, 102.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_xlabel("B1 prediction non-increase rate (%)", fontsize=15)
    axis.set_ylabel("B2 prediction non-increase rate (%)", fontsize=15)
    axis.set_title("Paired prediction non-increase rates", fontsize=17)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(color="#9CA3AF", alpha=0.22, linewidth=0.7)
    axis.legend(loc="upper left", frameon=False, fontsize=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/Figures/b1_b2_nonincrease_scatter.pdf"),
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("results/b1_b2_nonincrease_pairs.csv"),
    )
    args = parser.parse_args()

    pairs = collect_pairs(args.logs_dir)
    if not pairs:
        raise RuntimeError("No complete B1 and B2 pairs were found")
    write_pairs(args.pairs_csv, pairs)
    draw_figure(pairs, args.output)
    print(f"Wrote {len(pairs)} paired configurations to {args.output}")


if __name__ == "__main__":
    main()
