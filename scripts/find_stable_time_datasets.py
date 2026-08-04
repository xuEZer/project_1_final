#!/usr/bin/env python3
"""Rank TIME datasets by chronological distribution stability.

The script compares train/validation/test portions of raw variates before any
missing-value injection or model training. Lower scores indicate that the
value distribution and trend are more similar across chronological splits.

Examples:
  uv run python scripts/find_stable_time_datasets.py --local-files-only
  uv run python scripts/find_stable_time_datasets.py --datasets CPHL Coastal_T_S
  uv run python scripts/find_stable_time_datasets.py --top-k 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "scripts" / "time_metadata.sqlite"
DEFAULT_OUTPUT = ROOT / "results" / "time_stability"


@dataclass
class SeriesScore:
    dataset: str
    frequency: str
    item_id: str
    variate_index: int
    variate_name: str
    length: int
    missing_rate: float
    mean_shift: float
    std_shift: float
    quantile_shift: float
    wasserstein_shift: float
    trend_shift: float
    missing_shift: float
    stability_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find approximately stationary datasets in Real-TSF/TIME."
    )
    parser.add_argument("--repo-id", default="Real-TSF/TIME")
    parser.add_argument("--metadata-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="*", help="Dataset names to scan.")
    parser.add_argument("--frequencies", nargs="*", help="Frequencies to scan.")
    parser.add_argument("--min-length", type=int, default=1024)
    parser.add_argument("--max-missing-rate", type=float, default=0.30)
    parser.add_argument("--max-series-per-dataset", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download dataset files that are absent from the HF cache.",
    )
    args = parser.parse_args()
    if args.min_length < 64:
        parser.error("--min-length must be at least 64")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1:
        parser.error("split ratios must lie in (0, 1)")
    if args.val_ratio + args.test_ratio >= 0.8:
        parser.error("validation + test ratio must be below 0.8")
    if args.max_series_per_dataset < 1 or args.top_k < 1:
        parser.error("series limit and top-k must be positive")
    return args


def candidate_groups(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.metadata_db.exists():
        raise FileNotFoundError(
            f"Metadata database not found: {args.metadata_db}. Run "
            "scripts/collect_time_metadata.py first."
        )
    connection = sqlite3.connect(args.metadata_db)
    try:
        rows = connection.execute(
            """
            SELECT dataset, frequency, MAX(length), MIN(missing_rate)
            FROM variates
            GROUP BY dataset, frequency
            HAVING MAX(length) >= ? AND MIN(missing_rate) <= ?
            ORDER BY dataset, frequency
            """,
            (args.min_length, args.max_missing_rate),
        ).fetchall()
    finally:
        connection.close()
    groups = [
        (dataset, frequency)
        for dataset, frequency, _length, _missing in rows
        if (not args.datasets or dataset in args.datasets)
        and (not args.frequencies or frequency in args.frequencies)
    ]
    return groups[: args.max_datasets] if args.max_datasets else groups


def finite(segment: np.ndarray) -> np.ndarray:
    return np.asarray(segment[np.isfinite(segment)], dtype=np.float64)


def robust_scale(values: np.ndarray) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float(q75 - q25) / 1.349, float(np.std(values)), 1e-8)
    return scale


def trend_amplitude(segment: np.ndarray, scale: float) -> float:
    mask = np.isfinite(segment)
    if mask.sum() < 3:
        return math.nan
    time = np.linspace(0.0, 1.0, segment.size)[mask]
    slope = np.polyfit(time, segment[mask], 1)[0]
    return float(slope / scale)


def score_series(
    values: np.ndarray,
    dataset: str,
    frequency: str,
    item_id: str,
    variate_index: int,
    variate_name: str,
    val_ratio: float,
    test_ratio: float,
) -> SeriesScore | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))
    raw_parts = (values[:train_end], values[train_end:val_end], values[val_end:])
    parts = tuple(finite(part) for part in raw_parts)
    if min(map(len, parts)) < 32:
        return None

    train, val, test = parts
    scale = robust_scale(train)

    def maximum_against_train(function) -> float:
        return max(float(function(train, val)), float(function(train, test)))

    mean_shift = maximum_against_train(
        lambda left, right: abs(np.mean(right) - np.mean(left)) / scale
    )
    std_shift = maximum_against_train(
        lambda left, right: abs(
            math.log((np.std(right) + 1e-8) / (np.std(left) + 1e-8))
        )
    )
    quantiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    quantile_shift = maximum_against_train(
        lambda left, right: np.mean(
            np.abs(np.quantile(right, quantiles) - np.quantile(left, quantiles))
        ) / scale
    )
    wasserstein_shift = maximum_against_train(
        lambda left, right: wasserstein_distance(left, right) / scale
    )
    train_trend = trend_amplitude(raw_parts[0], scale)
    trend_shift = max(
        abs(trend_amplitude(raw_parts[1], scale) - train_trend),
        abs(trend_amplitude(raw_parts[2], scale) - train_trend),
    )
    missing_rates = [float(np.mean(~np.isfinite(part))) for part in raw_parts]
    missing_shift = max(
        abs(missing_rates[1] - missing_rates[0]),
        abs(missing_rates[2] - missing_rates[0]),
    )

    # The score is dimensionless. Distribution distance receives the largest
    # weight; trend and missingness prevent a deceptively stable marginal from
    # hiding strong temporal or observation-process changes.
    stability_score = (
        0.30 * wasserstein_shift
        + 0.20 * mean_shift
        + 0.15 * std_shift
        + 0.15 * quantile_shift
        + 0.15 * trend_shift
        + 0.05 * (5.0 * missing_shift)
    )
    names = (str(variate_name) if variate_name is not None else str(variate_index))
    return SeriesScore(
        dataset=dataset,
        frequency=frequency,
        item_id=str(item_id),
        variate_index=variate_index,
        variate_name=names,
        length=n,
        missing_rate=float(np.mean(~np.isfinite(values))),
        mean_shift=mean_shift,
        std_shift=std_shift,
        quantile_shift=quantile_shift,
        wasserstein_shift=wasserstein_shift,
        trend_shift=trend_shift,
        missing_shift=missing_shift,
        stability_score=stability_score,
    )


def evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit, dtype=int))


def scan_group(
    repo_id: str,
    dataset: str,
    frequency: str,
    args: argparse.Namespace,
) -> list[SeriesScore]:
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        allow_patterns=f"{dataset}/{frequency}/*",
        local_files_only=args.local_files_only,
    )
    collection = load_from_disk(f"{snapshot}/{dataset}/{frequency}")
    candidates: list[tuple[int, int]] = []
    for item_index, item in enumerate(collection):
        target = np.asarray(item["target"])
        variate_count = 1 if target.ndim == 1 else target.shape[0]
        candidates.extend((item_index, vi) for vi in range(variate_count))
    selected = evenly_spaced_indices(len(candidates), args.max_series_per_dataset)

    scores: list[SeriesScore] = []
    for candidate_index in selected:
        item_index, variate_index = candidates[int(candidate_index)]
        item = collection[item_index]
        target = np.asarray(item["target"], dtype=np.float64)
        values = target if target.ndim == 1 else target[variate_index]
        if values.size < args.min_length:
            continue
        if np.mean(~np.isfinite(values)) > args.max_missing_rate:
            continue
        names = item.get("variate_names")
        variate_name = (
            names[variate_index]
            if names is not None and variate_index < len(names)
            else str(variate_index)
        )
        result = score_series(
            values,
            dataset,
            frequency,
            item.get("item_id", item_index),
            variate_index,
            variate_name,
            args.val_ratio,
            args.test_ratio,
        )
        if result is not None:
            scores.append(result)
    return scores


def summarize(scores: list[SeriesScore]) -> list[dict]:
    groups: dict[tuple[str, str], list[SeriesScore]] = {}
    for score in scores:
        groups.setdefault((score.dataset, score.frequency), []).append(score)
    summaries = []
    for (dataset, frequency), rows in groups.items():
        values = np.array([row.stability_score for row in rows])
        representative = min(rows, key=lambda row: row.stability_score)
        summaries.append(
            {
                "dataset": dataset,
                "frequency": frequency,
                "series_scanned": len(rows),
                "median_score": float(np.median(values)),
                "p90_score": float(np.quantile(values, 0.90)),
                "ranking_score": float(
                    np.median(values) + 0.5 * np.quantile(values, 0.90)
                ),
                "representative_item_id": representative.item_id,
                "representative_variate_index": representative.variate_index,
                "representative_variate_name": representative.variate_name,
                "representative_length": representative.length,
                "representative_score": representative.stability_score,
            }
        )
    return sorted(summaries, key=lambda row: row["ranking_score"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    groups = candidate_groups(args)
    print(f"Scanning {len(groups)} TIME dataset-frequency groups...")
    all_scores: list[SeriesScore] = []
    failures: list[dict[str, str]] = []
    for index, (dataset, frequency) in enumerate(groups, start=1):
        label = f"{dataset}/{frequency}"
        try:
            rows = scan_group(args.repo_id, dataset, frequency, args)
            all_scores.extend(rows)
            print(f"[{index:>3}/{len(groups)}] {label}: {len(rows)} series")
        except Exception as error:  # continue the repository-wide scan
            failures.append({"dataset": dataset, "frequency": frequency,
                             "error": f"{type(error).__name__}: {error}"})
            print(f"[{index:>3}/{len(groups)}] {label}: SKIP ({error})")

    summaries = summarize(all_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "series_scores.csv", [asdict(x) for x in all_scores])
    write_csv(args.output_dir / "dataset_ranking.csv", summaries)
    write_csv(args.output_dir / "failures.csv", failures)
    report = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "groups_requested": len(groups),
        "groups_scored": len(summaries),
        "series_scored": len(all_scores),
        "failures": failures,
        "top": summaries[: args.top_k],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop stable TIME datasets (lower is better):")
    print("rank  dataset/frequency                 score    median   p90   representative")
    for rank, row in enumerate(summaries[: args.top_k], start=1):
        label = f"{row['dataset']}/{row['frequency']}"
        representative = (
            f"{row['representative_item_id']}:"
            f"{row['representative_variate_index']}"
        )
        print(
            f"{rank:>4}  {label:<34} {row['ranking_score']:>7.3f}  "
            f"{row['median_score']:>7.3f} {row['p90_score']:>6.3f}  "
            f"{representative}"
        )
    print(f"\nResults written to {args.output_dir}")
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
