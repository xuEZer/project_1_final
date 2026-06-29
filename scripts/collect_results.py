#!/usr/bin/env python3
"""Collect experiment metrics from logs into CSV files.

Outputs:
  results/experiment_results.csv
  results/relative_to_b1.csv
  results/summary_by_dataset.csv
  results/stat_tests.txt
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

METRICS = {"MAE", "MSE", "MRE", "RMSE"}
TIP_STATS_RE = re.compile(
    r"TIP stats \| variant=(?P<variant>\S+) \| steps=(?P<steps>\d+) \| "
    r"conflict_rate=(?P<conflict_rate>[-+0-9.eE]+) \| "
    r"projection_rate=(?P<projection_rate>[-+0-9.eE]+) \| "
    r"mean_dot=(?P<mean_dot>[-+0-9.eE]+) \| "
    r"mean_gp_norm=(?P<mean_gp_norm>[-+0-9.eE]+)"
)


def to_float(value: str | None) -> float | None:
    if value is None or value == "" or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def log_message(line: str) -> str:
    marker = " - INFO - "
    if marker in line:
        return line.split(marker, 1)[1].rstrip()
    return line.rstrip()


def parse_log(path: Path) -> dict[str, str | float | bool | None]:
    row: dict[str, str | float | bool | None] = {
        "log_file": str(path.relative_to(ROOT)),
        "exp_name": path.stem,
        "completed": False,
        "exp_type": None,
        "dataset": None,
        "missing_rate": None,
        "artificial_missing_rate": None,
        "lookback_len": None,
        "block_size": None,
        "imputator": None,
        "predictor": None,
        "seed": None,
        "tip_k": None,
        "tip_epsilon": None,
        "tip_pretrain_epochs": None,
        "imp_mae": None,
        "pred_mae": None,
        "imp_mse": None,
        "pred_mse": None,
        "imp_mre": None,
        "pred_mre": None,
        "imp_rmse": None,
        "pred_rmse": None,
    }

    section: str | None = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    row["completed"] = any("Experiment completed successfully!" in ln for ln in lines)

    for idx, line in enumerate(lines):
        msg = log_message(line)
        stripped = msg.strip()

        if stripped.startswith("Exp name:"):
            row["exp_name"] = stripped.split(":", 1)[1].strip()
            continue

        if re.fullmatch(r"\[[A-Za-z0-9_]+\]", stripped):
            section = stripped.strip("[]")
            continue

        if section and " = " in stripped:
            key, value = stripped.split(" = ", 1)
            if section == "experiment":
                if key == "exp_type":
                    row["exp_type"] = value
                elif key == "random_seed":
                    row["seed"] = value
            elif section == "dataset":
                if key in {
                    "dataset",
                    "missing_rate",
                    "artificial_missing_rate",
                    "lookback_len",
                    "block_size",
                }:
                    row[key] = value
            elif section == "imputator" and key == "model_type":
                row["imputator"] = value
            elif section == "predictor" and key == "model_type":
                row["predictor"] = value
            elif section == "training":
                if key in {"tip_k", "tip_epsilon", "tip_pretrain_epochs"}:
                    row[key] = value

        if "Evaluation Results Summary:" in msg:
            for metric_line in lines[idx + 1 : idx + 14]:
                metric_msg = log_message(metric_line).strip()
                parts = metric_msg.split()
                if not parts or parts[0] not in METRICS:
                    continue
                metric = parts[0].lower()
                if len(parts) >= 3:
                    row[f"imp_{metric}"] = to_float(parts[-2])
                    row[f"pred_{metric}"] = to_float(parts[-1])
                elif len(parts) == 2:
                    row[f"pred_{metric}"] = to_float(parts[-1])

    # Fallback for very old logs that did not print a full config section.
    if row["exp_type"] is None:
        row["exp_type"] = str(row["exp_name"]).split("_", 1)[0]

    return row


def parse_tip_stats(path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    exp_name = path.stem
    epoch = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        msg = log_message(line)
        match_epoch = re.search(r"Epoch\s+(\d+)/", msg)
        if match_epoch:
            epoch = int(match_epoch.group(1))
        match = TIP_STATS_RE.search(msg)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "exp_name": exp_name,
                "log_file": str(path.relative_to(ROOT)),
                "epoch": epoch,
                "variant": item["variant"],
                "steps": int(item["steps"]),
                "conflict_rate": float(item["conflict_rate"]),
                "projection_rate": float(item["projection_rate"]),
                "mean_dot": float(item["mean_dot"]),
                "mean_gp_norm": float(item["mean_gp_norm"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_key(row: dict) -> tuple:
    return (
        row.get("dataset"),
        row.get("missing_rate"),
        row.get("lookback_len"),
        row.get("imputator"),
        row.get("predictor"),
        row.get("seed"),
    )


def collect_relative(rows: list[dict]) -> list[dict]:
    baselines: dict[tuple, dict] = {}
    for row in rows:
        if row.get("exp_type") == "B1" and row.get("pred_mae") is not None:
            baselines[group_key(row)] = row

    rel_rows: list[dict] = []
    for row in rows:
        if row.get("exp_type") == "B1":
            continue
        base = baselines.get(group_key(row))
        if not base:
            continue
        pred_mae = row.get("pred_mae")
        base_pred_mae = base.get("pred_mae")
        pred_mse = row.get("pred_mse")
        base_pred_mse = base.get("pred_mse")
        if not isinstance(pred_mae, float) or not isinstance(base_pred_mae, float):
            continue
        rel_rows.append(
            {
                "exp_type": row.get("exp_type"),
                "dataset": row.get("dataset"),
                "missing_rate": row.get("missing_rate"),
                "lookback_len": row.get("lookback_len"),
                "imputator": row.get("imputator"),
                "predictor": row.get("predictor"),
                "seed": row.get("seed"),
                "pred_mae": pred_mae,
                "b1_pred_mae": base_pred_mae,
                "pred_mae_ratio": pred_mae / base_pred_mae
                if base_pred_mae
                else math.nan,
                "pred_mse": pred_mse,
                "b1_pred_mse": base_pred_mse,
                "pred_mse_ratio": pred_mse / base_pred_mse
                if isinstance(pred_mse, float)
                and isinstance(base_pred_mse, float)
                and base_pred_mse
                else math.nan,
            }
        )
    return rel_rows


def collect_summary(rel_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rel_rows:
        ratio = row.get("pred_mae_ratio")
        if isinstance(ratio, float) and not math.isnan(ratio):
            grouped[(row.get("dataset"), row.get("exp_type"))].append(ratio)

    out: list[dict] = []
    for (dataset, exp_type), values in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "exp_type": exp_type,
                "count": len(values),
                "mean_pred_mae_ratio": statistics.fmean(values),
                "median_pred_mae_ratio": statistics.median(values),
                "std_pred_mae_ratio": statistics.pstdev(values)
                if len(values) > 1
                else 0.0,
                "win_rate_vs_b1": sum(v < 1.0 for v in values) / len(values),
            }
        )
    return out


def stat_tests(rows: list[dict]) -> str:
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover - depends on environment
        return f"scipy unavailable, skipped statistical tests: {exc}\n"

    lines: list[str] = []

    for exp_type in ("B1", "B2"):
        points = [
            (row.get("imp_mae"), row.get("pred_mae"))
            for row in rows
            if row.get("exp_type") == exp_type
            and isinstance(row.get("imp_mae"), float)
            and isinstance(row.get("pred_mae"), float)
        ]
        if len(points) >= 3:
            imp, pred = zip(*points)
            rho, pvalue = stats.spearmanr(imp, pred)
            lines.append(
                f"Spearman {exp_type}: rho={rho:.6g}, p={pvalue:.6g}, n={len(points)}"
            )

    by_key_type: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        pred_mae = row.get("pred_mae")
        exp_type = row.get("exp_type")
        if isinstance(pred_mae, float) and isinstance(exp_type, str):
            by_key_type[group_key(row)][exp_type] = pred_mae

    friedman_types = ["B1", "B2", "B4", "B5", "B6"]
    aligned = [
        values
        for values in by_key_type.values()
        if all(exp_type in values for exp_type in friedman_types)
    ]
    if len(aligned) >= 2:
        samples = [[values[exp_type] for values in aligned] for exp_type in friedman_types]
        stat, pvalue = stats.friedmanchisquare(*samples)
        lines.append(
            f"Friedman pred MAE ({','.join(friedman_types)}): "
            f"chi2={stat:.6g}, p={pvalue:.6g}, n={len(aligned)}"
        )

        ranks_by_type = {exp_type: [] for exp_type in friedman_types}
        for values in aligned:
            ranked = stats.rankdata([values[exp_type] for exp_type in friedman_types])
            for exp_type, rank in zip(friedman_types, ranked, strict=True):
                ranks_by_type[exp_type].append(float(rank))
        avg_ranks = {
            exp_type: statistics.fmean(ranks)
            for exp_type, ranks in ranks_by_type.items()
        }
        lines.append(
            "Average ranks (lower is better): "
            + ", ".join(f"{k}={v:.3f}" for k, v in avg_ranks.items())
        )

    pairs = [
        (values["B1"], values["B2"])
        for values in by_key_type.values()
        if "B1" in values and "B2" in values
    ]
    if pairs:
        wins = sum(b2 < b1 for b1, b2 in pairs)
        ties = sum(b2 == b1 for b1, b2 in pairs)
        losses = sum(b2 > b1 for b1, b2 in pairs)
        effective = wins + losses
        if effective > 0:
            pvalue = stats.binomtest(wins, effective, 0.5, alternative="greater").pvalue
            lines.append(
                f"B2 vs B1 sign test: wins={wins}, ties={ties}, losses={losses}, "
                f"p={pvalue:.6g}"
            )

    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    log_paths = sorted(LOG_DIR.glob("*.log"))
    rows = [parse_log(path) for path in log_paths]

    fields = [
        "log_file",
        "completed",
        "exp_name",
        "exp_type",
        "dataset",
        "missing_rate",
        "artificial_missing_rate",
        "lookback_len",
        "block_size",
        "imputator",
        "predictor",
        "seed",
        "tip_k",
        "tip_epsilon",
        "tip_pretrain_epochs",
        "imp_mae",
        "pred_mae",
        "imp_mse",
        "pred_mse",
        "imp_mre",
        "pred_mre",
        "imp_rmse",
        "pred_rmse",
    ]
    write_csv(RESULT_DIR / "experiment_results.csv", rows, fields)

    rel_rows = collect_relative(rows)
    rel_fields = [
        "exp_type",
        "dataset",
        "missing_rate",
        "lookback_len",
        "imputator",
        "predictor",
        "seed",
        "pred_mae",
        "b1_pred_mae",
        "pred_mae_ratio",
        "pred_mse",
        "b1_pred_mse",
        "pred_mse_ratio",
    ]
    write_csv(RESULT_DIR / "relative_to_b1.csv", rel_rows, rel_fields)

    summary_rows = collect_summary(rel_rows)
    summary_fields = [
        "dataset",
        "exp_type",
        "count",
        "mean_pred_mae_ratio",
        "median_pred_mae_ratio",
        "std_pred_mae_ratio",
        "win_rate_vs_b1",
    ]
    write_csv(RESULT_DIR / "summary_by_dataset.csv", summary_rows, summary_fields)

    stats_text = stat_tests(rows)
    (RESULT_DIR / "stat_tests.txt").write_text(stats_text, encoding="utf-8")

    tip_rows = [item for path in log_paths for item in parse_tip_stats(path)]
    tip_fields = [
        "exp_name",
        "log_file",
        "epoch",
        "variant",
        "steps",
        "conflict_rate",
        "projection_rate",
        "mean_dot",
        "mean_gp_norm",
    ]
    write_csv(RESULT_DIR / "tip_epoch_stats.csv", tip_rows, tip_fields)

    print(f"Wrote {len(rows)} rows to {RESULT_DIR / 'experiment_results.csv'}")
    print(f"Wrote {len(rel_rows)} rows to {RESULT_DIR / 'relative_to_b1.csv'}")
    print(f"Wrote summary to {RESULT_DIR / 'summary_by_dataset.csv'}")
    print(f"Wrote statistical tests to {RESULT_DIR / 'stat_tests.txt'}")
    print(f"Wrote {len(tip_rows)} TIP stat rows to {RESULT_DIR / 'tip_epoch_stats.csv'}")


if __name__ == "__main__":
    main()
