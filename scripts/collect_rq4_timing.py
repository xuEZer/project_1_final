#!/usr/bin/env python3
"""Summarize RQ4 timing records and draw B1-relative cost-effect curves."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results" / "rq4_timing"

EPOCH_RE = re.compile(
    r"Epoch wall clock: phase=rq4 \| epoch=(?P<epoch>\d+) \| "
    r"wall_seconds=(?P<seconds>[\d.eE+-]+)"
)
BASE_RE = re.compile(
    r"Counterfactual base step timing: phase=rq4 \| epoch=(?P<epoch>\d+) \| "
    r"step=(?P<step>\d+) \| wall_seconds=(?P<seconds>[\d.eE+-]+)"
)
TIP_RE = re.compile(
    r"RQ4 TIP timing: epoch=(?P<epoch>\d+) \| step=(?P<step>\d+) \| "
    r"rank=(?P<rank>\d+) \| wall_seconds=(?P<seconds>[\d.eE+-]+)"
)
EFFECT_RE = re.compile(
    r"RQ4 counterfactual step: step=(?P<step>\d+) \| "
    r"sweep=(?P<sweep>B1|rank) \| value=(?P<value>[^| ]+) \| "
    r"pred_loss_before=[-+\d.eE]+ \| "
    r"candidate_pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"conflict=(?P<conflict>[01]) \| projected=(?P<projected>[01]) \| "
    r"gain_vs_raw=(?P<gain>[-+\d.eE]+) \| "
    r"relative_gain=[-+\d.eE]+"
)
SEED_RE = re.compile(
    r"Seed routing: experiment_seed=(?P<experiment>\d+) \| "
    r"missing_seed=(?P<missing>\d+)"
)
UID_RE = re.compile(r"Generated preprocess UID: (?P<uid>[0-9a-f]{16})")


def mean_std(values: list[float]) -> tuple[float, float]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), std


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pareto_mask(rows: list[dict]) -> list[bool]:
    flags = []
    for row in rows:
        dominated = any(
            other["relative_step_time_multiplier"]
            <= row["relative_step_time_multiplier"]
            and other["intervention_improve_rate"]
            >= row["intervention_improve_rate"]
            and (
                other["relative_step_time_multiplier"]
                < row["relative_step_time_multiplier"]
                or other["intervention_improve_rate"]
                > row["intervention_improve_rate"]
            )
            for other in rows
        )
        flags.append(not dominated)
    return flags


def per_seed_means(values: dict[tuple[int, int], float]) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for (seed, _), value in values.items():
        grouped[seed].append(value)
    return [statistics.fmean(grouped[seed]) for seed in sorted(grouped)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir / "summary"
    )
    logs = sorted(input_dir.glob("seed_*/logs/RQ4_*.log"))
    if not logs:
        raise FileNotFoundError(f"No RQ4 timing logs found under {input_dir}")

    epoch_by_number: dict[int, list[float]] = defaultdict(list)
    base_by_key: dict[tuple[int, int], float] = {}
    tip_by_rank: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    effects: dict[str, dict[int, list[dict[str, float | bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    preprocess_uids: set[str] = set()
    observed_seeds: set[int] = set()

    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        seed = int(path.parents[1].name.split("_")[-1])
        seed_match = SEED_RE.search(text)
        uid_match = UID_RE.search(text)
        if seed_match is None or uid_match is None:
            raise ValueError(f"Seed routing metadata is missing from {path}")
        if int(seed_match.group("experiment")) != seed:
            raise ValueError(f"Experiment seed does not match directory for {path}")
        if int(seed_match.group("missing")) != 42:
            raise ValueError(f"Missing-data seed is not 42 in {path}")
        observed_seeds.add(seed)
        preprocess_uids.add(uid_match.group("uid"))
        for match in EPOCH_RE.finditer(text):
            epoch_by_number[int(match.group("epoch"))].append(float(match.group("seconds")))
        for match in BASE_RE.finditer(text):
            key = (seed, int(match.group("step")))
            base_by_key[key] = float(match.group("seconds"))
        for match in TIP_RE.finditer(text):
            key = (seed, int(match.group("step")))
            tip_by_rank[int(match.group("rank"))][key] = float(match.group("seconds"))
        for match in EFFECT_RE.finditer(text):
            label = (
                "B1"
                if match.group("sweep") == "B1"
                else f"TIP-r{match.group('value')}"
            )
            effects[label][seed].append(
                {
                    "nonincrease": float(match.group("delta")) <= 0.0,
                    "projected": bool(int(match.group("projected"))),
                    "gain": float(match.group("gain")),
                }
            )

    if observed_seeds != {42, 43, 44}:
        raise ValueError(
            f"Expected timing seeds 42, 43, 44; found {sorted(observed_seeds)}"
        )
    if len(preprocess_uids) != 1:
        raise ValueError(
            "Timing runs do not share one fixed missing-data preprocessing UID"
        )

    epoch_rows = []
    for epoch, values in sorted(epoch_by_number.items()):
        mean, std = mean_std(values)
        epoch_rows.append(
            {
                "epoch": epoch,
                "runs": len(values),
                "wall_time_mean_s": mean,
                "wall_time_std_s": std,
            }
        )
    if not epoch_rows:
        raise ValueError("No epoch wall-clock records were found")
    write_csv(output_dir / "epoch_wall_clock.csv", epoch_rows)

    if not base_by_key or not tip_by_rank:
        raise ValueError("Timing records are incomplete; rerun RQ4 with timing instrumentation")
    base_run_means = per_seed_means(base_by_key)
    base_mean, base_std = mean_std(base_run_means)
    rows = []
    b1_effect = effects.get("B1", {})
    if not b1_effect:
        raise ValueError("No B1 prediction-effect records were found")
    b1_effect_rates = [
        100.0 * sum(item["nonincrease"] for item in values) / len(values)
        for _, values in sorted(b1_effect.items())
    ]
    b1_effect_mean, b1_effect_std = mean_std(b1_effect_rates)
    rows.append(
        {
            "method": "B1",
            "rank": "",
            "seeds": len(base_run_means),
            "step_observations": len(base_by_key),
            "wall_time_mean_ms": 1000.0 * base_mean,
            "wall_time_std_ms": 1000.0 * base_std,
            "tip_overhead_mean_ms": 0.0,
            "tip_overhead_std_ms": 0.0,
            "prediction_nonincrease_rate": b1_effect_mean,
            "prediction_nonincrease_std": b1_effect_std,
            "relative_step_time_multiplier": 1.0,
            "relative_step_time_multiplier_std": 0.0,
            "intervention_improve_rate": "",
            "intervention_improve_std": "",
        }
    )
    for rank, tip_by_key in sorted(tip_by_rank.items()):
        if tip_by_key.keys() != base_by_key.keys():
            raise ValueError(f"B1 and rank-{rank} timing records do not align")
        ordered_keys = sorted(base_by_key)
        tip_run_means = per_seed_means(tip_by_key)
        tip_mean, tip_std = mean_std(tip_run_means)
        total_by_key = {
            key: base_by_key[key] + tip_by_key[key] for key in ordered_keys
        }
        total_run_means = per_seed_means(total_by_key)
        total_mean, total_std = mean_std(total_run_means)
        label = f"TIP-r{rank}"
        effect_values = effects.get(label, {})
        if not effect_values:
            raise ValueError(f"No prediction-effect records were found for rank {rank}")
        effect_rates = [
            100.0 * sum(item["nonincrease"] for item in values) / len(values)
            for _, values in sorted(effect_values.items())
        ]
        effect_mean, effect_std = mean_std(effect_rates)
        time_multipliers = [
            (base_by_key[key] + tip_by_key[key]) / base_by_key[key]
            for key in ordered_keys
        ]
        time_multiplier_by_seed = per_seed_means(
            {
                key: value
                for key, value in zip(ordered_keys, time_multipliers, strict=True)
            }
        )
        time_multiplier_mean, time_multiplier_std = mean_std(time_multiplier_by_seed)
        intervention_improve_rates = [
            100.0
            * sum(item["gain"] > 0.0 for item in values if item["projected"])
            / sum(item["projected"] for item in values)
            for _, values in sorted(effect_values.items())
        ]
        intervention_improve_mean, intervention_improve_std = mean_std(
            intervention_improve_rates
        )
        rows.append(
            {
                "method": "TIP",
                "rank": rank,
                "seeds": len(tip_run_means),
                "step_observations": len(tip_by_key),
                "wall_time_mean_ms": 1000.0 * total_mean,
                "wall_time_std_ms": 1000.0 * total_std,
                "tip_overhead_mean_ms": 1000.0 * tip_mean,
                "tip_overhead_std_ms": 1000.0 * tip_std,
                "prediction_nonincrease_rate": effect_mean,
                "prediction_nonincrease_std": effect_std,
                "relative_step_time_multiplier": time_multiplier_mean,
                "relative_step_time_multiplier_std": time_multiplier_std,
                "intervention_improve_rate": intervention_improve_mean,
                "intervention_improve_std": intervention_improve_std,
            }
        )

    tip_rows = [row for row in rows if row["method"] == "TIP"]
    flags = pareto_mask(tip_rows)
    for row in rows:
        row["pareto_optimal"] = 0
    for row, flag in zip(tip_rows, flags, strict=True):
        row["pareto_optimal"] = int(flag)
    write_csv(output_dir / "time_effect_summary.csv", rows)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for row, flag in zip(tip_rows, flags, strict=True):
        label = f"r={row['rank']}"
        axis.errorbar(
            row["relative_step_time_multiplier"],
            row["intervention_improve_rate"],
            xerr=row["relative_step_time_multiplier_std"],
            yerr=row["intervention_improve_std"],
            fmt="o",
            capsize=3,
            color="tab:red" if flag else "tab:blue",
        )
        axis.annotate(
            label,
            (
                row["relative_step_time_multiplier"],
                row["intervention_improve_rate"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )
    frontier = sorted(
        (row for row, flag in zip(tip_rows, flags, strict=True) if flag),
        key=lambda item: item["relative_step_time_multiplier"],
    )
    axis.plot(
        [row["relative_step_time_multiplier"] for row in frontier],
        [row["intervention_improve_rate"] for row in frontier],
        color="tab:red",
        linewidth=1.2,
    )
    axis.set_xlabel("Relative step-time multiplier vs. B1")
    axis.set_ylabel("Intervention improvement rate (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "time_effect_pareto.pdf")
    figure.savefig(output_dir / "time_effect_pareto.png", dpi=240)
    plt.close(figure)
    print(f"Summarized {len(logs)} logs into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
