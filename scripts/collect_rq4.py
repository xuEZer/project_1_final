#!/usr/bin/env python3
"""Collect RQ4 shared-trajectory sensitivity metrics into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STEP_RE = re.compile(
    r"RQ4 counterfactual step: step=(?P<step>\d+) \| "
    r"sweep=(?P<sweep>B1|rank|epsilon|gate_delta) \| value=(?P<value>[^| ]+) \| "
    r"pred_loss_before=(?P<before>[-+\d.eE]+) \| "
    r"candidate_pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"conflict=(?P<conflict>[01]) \| projected=(?P<projected>[01]) \| "
    r"gain_vs_raw=(?P<gain>[-+\d.eE]+) \| "
    r"relative_gain=(?P<relative>[-+\d.eE]+)"
)
SEED_RE = re.compile(
    r"Seed routing: experiment_seed=(?P<experiment>\d+) \| "
    r"missing_seed=(?P<missing>\d+)"
)
UID_RE = re.compile(r"Generated preprocess UID: (?P<uid>[0-9a-f]{16})")


def completed_seed_logs(logs_dir: Path) -> dict[int, Path]:
    """Return the complete three-seed RQ4 log set with one shared mask UID."""
    paths: dict[int, Path] = {}
    for path in logs_dir.glob("RQ4_azure2019_I_5T_SAITS_Toto2_*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Experiment completed successfully!" not in text:
            continue
        seed_match = SEED_RE.search(text)
        uid_match = UID_RE.search(text)
        if seed_match is None or uid_match is None:
            continue
        seed = int(seed_match.group("experiment"))
        if int(seed_match.group("missing")) != 42 or seed not in {42, 43, 44}:
            continue
        previous = paths.get(seed)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            paths[seed] = path
    if set(paths) != {42, 43, 44}:
        raise FileNotFoundError(
            f"Expected completed RQ4 seeds 42, 43, 44; found {sorted(paths)}"
        )
    selected_uids = {
        UID_RE.search(path.read_text(encoding="utf-8", errors="replace")).group("uid")
        for path in paths.values()
    }
    if len(selected_uids) != 1:
        raise ValueError("RQ4 seed logs do not share one preprocessing UID")
    return paths


def latest_completed(logs_dir: Path) -> Path:
    candidates = []
    for path in logs_dir.glob("RQ4_azure2019_I_5T_SAITS_Toto2_*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Experiment completed successfully!" in text:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed RQ4 log was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect(path: Path) -> list[dict]:
    by_step: dict[int, dict[tuple[str, str], dict]] = defaultdict(dict)
    for match in STEP_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        item = {
            "delta": float(match.group("delta")),
            "conflict": int(match.group("conflict")),
            "projected": int(match.group("projected")),
            "gain": float(match.group("gain")),
            "relative": float(match.group("relative")),
        }
        by_step[int(match.group("step"))][
            (match.group("sweep"), match.group("value"))
        ] = item
    if not by_step:
        raise ValueError(f"No RQ4 step records found in {path}")

    safe_steps = [
        variants
        for variants in by_step.values()
        if variants.get(("B1", "base"), {}).get("delta", float("inf")) <= 0.0
    ]
    harmful_steps = [
        variants
        for variants in by_step.values()
        if variants.get(("B1", "base"), {}).get("delta", float("-inf")) > 0.0
    ]
    keys = sorted({key for variants in by_step.values() for key in variants})
    rows = []
    for sweep, value in keys:
        items = [variants[(sweep, value)] for variants in by_step.values()]
        projected = [item for item in items if item["projected"]]
        row = {
            "sweep": sweep,
            "value": value,
            "steps": len(items),
            "prediction_nonincrease_rate": 100.0 * sum(
                item["delta"] <= 0.0 for item in items
            ) / len(items),
            "conflict_rate": 100.0 * sum(item["conflict"] for item in items) / len(items),
            "projection_rate": 100.0 * len(projected) / len(items),
            "false_positive_rate": "",
            "true_positive_rate": "",
            "intervention_improve_rate": "",
            "relative_gain_mean": "",
            "relative_gain_std": "",
        }
        if sweep != "B1":
            row["false_positive_rate"] = 100.0 * sum(
                variants[(sweep, value)]["conflict"] for variants in safe_steps
            ) / len(safe_steps)
            row["true_positive_rate"] = 100.0 * sum(
                variants[(sweep, value)]["conflict"] for variants in harmful_steps
            ) / len(harmful_steps)
            if projected:
                gains = [100.0 * item["relative"] for item in projected]
                row["intervention_improve_rate"] = 100.0 * sum(
                    item["gain"] > 0.0 for item in projected
                ) / len(projected)
                row["relative_gain_mean"] = statistics.fmean(gains)
                row["relative_gain_std"] = statistics.pstdev(gains)
        rows.append(row)
    return rows


def aggregate(paths: dict[int, Path]) -> list[dict]:
    """Aggregate seed-level RQ4 rates and report variation across seeds."""
    seed_rows = {seed: collect(path) for seed, path in paths.items()}
    lookups = {
        seed: {(row["sweep"], row["value"]): row for row in rows}
        for seed, rows in seed_rows.items()
    }
    keys = sorted(set.intersection(*(set(rows) for rows in lookups.values())))
    output = []
    rate_fields = (
        "prediction_nonincrease_rate",
        "conflict_rate",
        "projection_rate",
        "false_positive_rate",
        "true_positive_rate",
        "intervention_improve_rate",
    )
    for key in keys:
        rows = [lookups[seed][key] for seed in sorted(lookups)]
        row = {
            "sweep": key[0],
            "value": key[1],
            "seeds": len(rows),
            "steps": sum(item["steps"] for item in rows),
        }
        for field in rate_fields:
            values = [float(item[field]) for item in rows if item[field] != ""]
            row[field] = statistics.fmean(values) if values else ""
        gain_means = [
            float(item["relative_gain_mean"])
            for item in rows
            if item["relative_gain_mean"] != ""
        ]
        row["relative_gain_mean"] = (
            statistics.fmean(gain_means) if gain_means else ""
        )
        row["relative_gain_std"] = (
            statistics.stdev(gain_means) if len(gain_means) > 1 else ""
        )
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "figures" / "rq4_sensitivity.csv",
    )
    args = parser.parse_args()
    paths = completed_seed_logs(args.logs_dir)
    rows = aggregate(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Collected {len(rows)} configurations from seeds "
        f"{sorted(paths)} into {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
