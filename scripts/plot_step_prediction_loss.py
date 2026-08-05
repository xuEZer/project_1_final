#!/usr/bin/env python3
"""Plot per-optimizer-step prediction losses from paired B1/B2 logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEP_PATTERN = re.compile(
    r"Prediction optimizer step: step=(?P<step>\d+) \| "
    r"pred_loss_before=(?P<before>[-+\d.eE]+) \| "
    r"raw_adamw_pred_loss_delta=(?P<raw_delta>[-+\d.eE]+) \| "
    r"pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"pred_loss_relative_delta=(?P<relative_delta>[-+\d.eE]+) \| "
    r"tip_projection_gain=(?P<gain>[-+\d.eE]+)"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-log", required=True, type=Path)
    parser.add_argument("--b2-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional path for the aligned step-level source data.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=25,
        help="Trailing moving-average window (default: 25; 1 disables).",
    )
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    return args


def parse_log(path: Path) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_PATTERN.search(line)
        if match is None:
            continue
        values = match.groupdict()
        # ``after`` values are no longer logged (redundant with delta);
        # reconstruct them from before + delta for plotting.
        before = float(values["before"])
        raw_delta = float(values["raw_delta"])
        delta = float(values["delta"])
        records.append(
            {
                "step": int(values["step"]),
                "before": before,
                "raw_after": before + raw_delta,
                "after": before + delta,
                "raw_delta": raw_delta,
                "delta": delta,
                "relative_delta": float(values["relative_delta"]),
                "gain": float(values["gain"]),
            }
        )
    if not records:
        raise ValueError(f"No step-level prediction records found in {path}")
    return records


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        result[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return result


def align_records(
    b1: list[dict[str, float | int | str]],
    b2: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    by_step_b1 = {int(row["step"]): row for row in b1}
    by_step_b2 = {int(row["step"]): row for row in b2}
    shared_steps = sorted(by_step_b1.keys() & by_step_b2.keys())
    if not shared_steps:
        raise ValueError("B1 and B2 logs contain no shared optimizer steps")
    rows = []
    for step in shared_steps:
        left, right = by_step_b1[step], by_step_b2[step]
        rows.append(
            {
                "step": step,
                "b1_before": float(left["before"]),
                "b1_after": float(left["after"]),
                "b1_delta": float(left["delta"]),
                "b2_before": float(right["before"]),
                "b2_raw_after": float(right["raw_after"]),
                "b2_after": float(right["after"]),
                "b2_raw_delta": float(right["raw_delta"]),
                "b2_delta": float(right["delta"]),
                "b2_projection_gain": float(right["gain"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, float | int]],
    output: Path,
    smooth: int,
) -> None:
    step = np.asarray([row["step"] for row in rows], dtype=int)
    b1_after = np.asarray([row["b1_after"] for row in rows], dtype=float)
    b2_raw_after = np.asarray(
        [row["b2_raw_after"] for row in rows], dtype=float
    )
    b2_after = np.asarray([row["b2_after"] for row in rows], dtype=float)
    b1_delta = np.asarray([row["b1_delta"] for row in rows], dtype=float)
    b2_raw_delta = np.asarray(
        [row["b2_raw_delta"] for row in rows], dtype=float
    )
    b2_delta = np.asarray([row["b2_delta"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(step, moving_average(b1_after, smooth), label="B1 after")
    axes[0].plot(
        step, moving_average(b2_raw_after, smooth), linestyle="--",
        label="B2 raw AdamW after",
    )
    axes[0].plot(step, moving_average(b2_after, smooth), label="B2 TIP after")
    axes[0].set_ylabel("Current-batch prediction MSE")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(step, moving_average(b1_delta, smooth), label="B1 delta")
    axes[1].plot(
        step,
        moving_average(b2_raw_delta, smooth),
        linestyle="--",
        alpha=0.75,
        label="B2 raw AdamW delta",
    )
    axes[1].plot(step, moving_average(b2_delta, smooth), label="B2 TIP delta")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Prediction MSE change")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "Current-batch prediction trajectory and local step effect "
        f"(moving average={smooth})"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = align_records(parse_log(args.b1_log), parse_log(args.b2_log))
    plot(rows, args.output, args.smooth)
    if args.csv_output is not None:
        write_csv(args.csv_output, rows)
    print(
        f"Wrote {args.output} from {len(rows)} aligned optimizer steps"
        + (f" and {args.csv_output}" if args.csv_output is not None else "")
    )


if __name__ == "__main__":
    main()
