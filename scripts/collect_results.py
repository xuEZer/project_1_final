#!/usr/bin/env python3
"""Collect paper-suite experiment metrics with placeholders.

The training scripts only create log files after a job starts.  This collector
builds the experiment matrices used by scripts/run_paper_experiments.sh first,
then joins logs onto that plan so unfinished or unstarted jobs remain visible.

Outputs:
  results/experiment_results.csv
  results/main_experiment_results.csv
  results/projection_ablation_results.csv
  results/k_sweep_results.csv
  results/epsilon_sweep_results.csv
  results/missing_rate_robustness_results.csv
  results/compute_cost_probe_results.csv
  results/missing_experiments.csv
  results/unplanned_logs.csv
  results/relative_to_b1.csv
  results/summary_by_dataset.csv
  results/stat_tests.txt
  results/tip_epoch_stats.csv
"""

from __future__ import annotations

import argparse
import configparser
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
COMMON_SCRIPT = SCRIPT_DIR / "common_experiment.sh"
GLOBAL_CONFIG = ROOT / "configs" / "global.ini"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results"

METRICS = {"MAE", "MSE", "MRE", "RMSE"}
METRIC_FIELDS = [
    "imp_mae",
    "pred_mae",
    "imp_mse",
    "pred_mse",
    "imp_mre",
    "pred_mre",
    "imp_rmse",
    "pred_rmse",
]
CONFIG_FIELDS = [
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
]
RESULT_FIELDS = [
    "suite",
    "suite_order",
    "status",
    "completed",
    "expected_exp_name",
    "exp_name",
    "log_file",
    *CONFIG_FIELDS,
    *METRIC_FIELDS,
    "elapsed_seconds",
    "exit_code",
]
TIP_STATS_RE = re.compile(
    r"TIP stats \| variant=(?P<variant>\S+) \| steps=(?P<steps>\d+) \| "
    r"conflict_rate=(?P<conflict_rate>[-+0-9.eE]+) \| "
    r"projection_rate=(?P<projection_rate>[-+0-9.eE]+) \| "
    r"mean_dot=(?P<mean_dot>[-+0-9.eE]+) \| "
    r"mean_gp_norm=(?P<mean_gp_norm>[-+0-9.eE]+)"
)

SUITE_ORDER = {
    "main_experiment": 1,
    "projection_ablation": 2,
    "k_sweep": 3,
    "epsilon_sweep": 4,
    "missing_rate_robustness": 5,
    "compute_cost_probe": 6,
    "unplanned": 99,
}
SUITE_OUTPUTS = {
    "main_experiment": "main_experiment_results.csv",
    "projection_ablation": "projection_ablation_results.csv",
    "k_sweep": "k_sweep_results.csv",
    "epsilon_sweep": "epsilon_sweep_results.csv",
    "missing_rate_robustness": "missing_rate_robustness_results.csv",
    "compute_cost_probe": "compute_cost_probe_results.csv",
}
STATUS_SCORE = {"missing_log": 0, "incomplete": 1, "completed": 2}


@dataclass(frozen=True)
class CommonSettings:
    paper_datasets: list[str]
    paper_imputators: list[str]
    paper_predictors: list[str]
    paper_missing_rates: list[str]
    default_main_missing_rate: str
    default_lookback_len: str
    default_block_size: str
    default_seeds: list[str]
    default_tip_k: str
    default_tip_epsilon: str
    default_pretrain_epochs: str


@dataclass(frozen=True)
class PlannedExperiment:
    suite: str
    exp_type: str
    dataset: str
    imputator: str
    predictor: str
    missing_rate: str
    seed: str
    lookback_len: str
    block_size: str
    tip_k: str
    tip_epsilon: str
    tip_pretrain_epochs: str
    include_epsilon_in_name: bool = False
    exp_name_suffix: str = ""

    @property
    def suite_order(self) -> int:
        return SUITE_ORDER[self.suite]

    @property
    def expected_exp_name(self) -> str:
        exp_name = experiment_name(
            self.exp_type,
            self.dataset,
            self.imputator,
            self.predictor,
            self.missing_rate,
            self.lookback_len,
            self.seed,
            self.tip_k,
            self.tip_epsilon,
            self.include_epsilon_in_name,
        )
        if self.exp_name_suffix:
            exp_name = f"{exp_name}_{self.exp_name_suffix}"
        return exp_name


def fallback_settings() -> CommonSettings:
    config = configparser.ConfigParser()
    config.read(GLOBAL_CONFIG)

    def config_get(section: str, key: str, fallback: str) -> str:
        if not config.has_section(section):
            return fallback
        return config.get(section, key, fallback=fallback)

    return CommonSettings(
        paper_datasets=[
            "Supply_Chain_Customer_D",
            "current_velocity_H",
            "azure2019_D_5T",
            "Coastal_T_S_15T",
            "CPHL_30T",
            "MetroPT-3_5T",
            "Australia_Solar_H",
        ],
        paper_imputators=["BRITS", "SAITS", "DeepMVI", "TimesNet"],
        paper_predictors=["Sundial", "TimesFM25"],
        paper_missing_rates=["0.1", "0.2", "0.3", "0.4", "0.5"],
        default_main_missing_rate=config_get("dataset", "missing_rate", "0.2"),
        default_lookback_len=config_get("dataset", "lookback_len", "512"),
        default_block_size=config_get("dataset", "block_size", "20"),
        default_seeds=[config_get("experiment", "random_seed", "42")],
        default_tip_k=config_get("training", "tip_k", "20"),
        default_tip_epsilon=config_get("training", "tip_epsilon", "1e-4"),
        default_pretrain_epochs=config_get("training", "tip_pretrain_epochs", "2"),
    )


def strip_shell_value(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def is_dynamic_shell_value(value: str) -> bool:
    return "$" in value or "`" in value


def parse_shell_array(text: str, name: str) -> list[str] | None:
    match = re.search(rf"^{re.escape(name)}=\((?P<body>.*?)\)", text, re.M | re.S)
    if not match:
        return None
    body = "\n".join(line.split("#", 1)[0] for line in match.group("body").splitlines())
    items: list[str] = []
    for token in re.finditer(r'"([^"]*)"|\'([^\']*)\'|([^\s()]+)', body):
        value = token.group(1) or token.group(2) or token.group(3)
        if value:
            items.append(value)
    return items


def parse_shell_scalar(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(?P<value>[^\n]+)", text, re.M)
    if not match:
        return None
    value = strip_shell_value(match.group("value"))
    if is_dynamic_shell_value(value):
        return None
    return value


def static_shell_array(text: str, name: str) -> list[str] | None:
    values = parse_shell_array(text, name)
    if not values or any(is_dynamic_shell_value(value) for value in values):
        return None
    return values


def load_common_settings() -> CommonSettings:
    settings = fallback_settings()
    if not COMMON_SCRIPT.exists():
        return settings

    text = COMMON_SCRIPT.read_text(encoding="utf-8", errors="replace")

    return CommonSettings(
        paper_datasets=parse_shell_array(text, "PAPER_DATASETS")
        or settings.paper_datasets,
        paper_imputators=parse_shell_array(text, "PAPER_IMPUTATORS")
        or settings.paper_imputators,
        paper_predictors=parse_shell_array(text, "PAPER_PREDICTORS")
        or settings.paper_predictors,
        paper_missing_rates=parse_shell_array(text, "PAPER_MISSING_RATES")
        or settings.paper_missing_rates,
        default_main_missing_rate=parse_shell_scalar(text, "DEFAULT_MAIN_MISSING_RATE")
        or settings.default_main_missing_rate,
        default_lookback_len=parse_shell_scalar(text, "DEFAULT_LOOKBACK_LEN")
        or settings.default_lookback_len,
        default_block_size=parse_shell_scalar(text, "DEFAULT_BLOCK_SIZE")
        or settings.default_block_size,
        default_seeds=static_shell_array(text, "DEFAULT_SEEDS")
        or settings.default_seeds,
        default_tip_k=parse_shell_scalar(text, "DEFAULT_TIP_K")
        or settings.default_tip_k,
        default_tip_epsilon=parse_shell_scalar(text, "DEFAULT_TIP_EPSILON")
        or settings.default_tip_epsilon,
        default_pretrain_epochs=parse_shell_scalar(text, "DEFAULT_PRETRAIN_EPOCHS")
        or settings.default_pretrain_epochs,
    )


def csv_values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(settings: CommonSettings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect run_paper_experiments.sh results, keeping placeholder rows "
            "for missing or incomplete logs."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Match the reduced grids selected by run_paper_experiments.sh --quick.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(settings.default_seeds),
        help="Comma-separated random seeds. Default is read from common_experiment.sh.",
    )
    parser.add_argument(
        "--lookback",
        default=settings.default_lookback_len,
        help="Lookback length used by the paper suite.",
    )
    parser.add_argument(
        "--main-missing-rate",
        default=settings.default_main_missing_rate,
        help="Missing rate for main, projection, and sweep experiments.",
    )
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated k sweep values. Defaults to the paper or quick grid.",
    )
    parser.add_argument(
        "--epsilon-values",
        default=None,
        help="Comma-separated epsilon sweep values. Defaults to the paper or quick grid.",
    )
    parser.add_argument(
        "--missing-rates",
        default=None,
        help="Comma-separated missing-rate robustness values.",
    )
    parser.add_argument(
        "--compute-max-epoch",
        default="1",
        help="Max epoch suffix used by run_compute_cost_probe.sh.",
    )
    return parser.parse_args()


def eps_label(epsilon: str) -> str:
    return epsilon.replace("+", "p")


def experiment_name(
    exp_type: str,
    dataset: str,
    imputator: str,
    predictor: str,
    missing_rate: str,
    lookback_len: str,
    seed: str,
    tip_k: str,
    tip_epsilon: str,
    include_epsilon_in_name: bool,
) -> str:
    if exp_type == "B1":
        return (
            f"{exp_type}_{dataset}_{imputator}_{predictor}_"
            f"mr{missing_rate}_L{lookback_len}_s{seed}"
        )
    if include_epsilon_in_name:
        return (
            f"{exp_type}_{dataset}_{imputator}_{predictor}_"
            f"mr{missing_rate}_L{lookback_len}_k{tip_k}_"
            f"eps{eps_label(tip_epsilon)}_s{seed}"
        )
    return (
        f"{exp_type}_{dataset}_{imputator}_{predictor}_"
        f"mr{missing_rate}_L{lookback_len}_k{tip_k}_s{seed}"
    )


def add_grid(
    plans: list[PlannedExperiment],
    *,
    suite: str,
    exp_types: list[str],
    datasets: list[str],
    imputators: list[str],
    predictors: list[str],
    missing_rates: list[str],
    seeds: list[str],
    settings: CommonSettings,
    lookback_len: str,
    tip_k: str,
    tip_epsilon: str,
    include_epsilon_in_name: bool = False,
    exp_name_suffix: str = "",
) -> None:
    for exp_type in exp_types:
        for dataset in datasets:
            for imputator in imputators:
                for predictor in predictors:
                    for missing_rate in missing_rates:
                        for seed in seeds:
                            plans.append(
                                PlannedExperiment(
                                    suite=suite,
                                    exp_type=exp_type,
                                    dataset=dataset,
                                    imputator=imputator,
                                    predictor=predictor,
                                    missing_rate=missing_rate,
                                    seed=seed,
                                    lookback_len=lookback_len,
                                    block_size=settings.default_block_size,
                                    tip_k=tip_k,
                                    tip_epsilon=tip_epsilon,
                                    tip_pretrain_epochs=settings.default_pretrain_epochs,
                                    include_epsilon_in_name=include_epsilon_in_name,
                                    exp_name_suffix=exp_name_suffix,
                                )
                            )


def add_paired_configs(
    plans: list[PlannedExperiment],
    *,
    suite: str,
    exp_type: str,
    configs: list[tuple[str, str, str]],
    values: list[str],
    value_kind: str,
    seeds: list[str],
    settings: CommonSettings,
    missing_rate: str,
    lookback_len: str,
) -> None:
    for dataset, imputator, predictor in configs:
        for value in values:
            for seed in seeds:
                plans.append(
                    PlannedExperiment(
                        suite=suite,
                        exp_type=exp_type,
                        dataset=dataset,
                        imputator=imputator,
                        predictor=predictor,
                        missing_rate=missing_rate,
                        seed=seed,
                        lookback_len=lookback_len,
                        block_size=settings.default_block_size,
                        tip_k=value if value_kind == "k" else settings.default_tip_k,
                        tip_epsilon=(
                            value if value_kind == "epsilon" else settings.default_tip_epsilon
                        ),
                        tip_pretrain_epochs=settings.default_pretrain_epochs,
                        include_epsilon_in_name=value_kind == "epsilon",
                    )
                )


def build_plans(args: argparse.Namespace, settings: CommonSettings) -> list[PlannedExperiment]:
    seeds = csv_values(args.seeds) or settings.default_seeds
    lookback_len = str(args.lookback)
    main_missing_rate = str(args.main_missing_rate)
    k_values = csv_values(args.k_values) or (
        ["5", "20"] if args.quick else ["5", "10", "20", "50", "100"]
    )
    epsilon_values = csv_values(args.epsilon_values) or (
        ["1e-4", "1e-3"] if args.quick else ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
    )
    robustness_missing_rates = csv_values(args.missing_rates) or (
        ["0.2", "0.4"] if args.quick else settings.paper_missing_rates
    )

    plans: list[PlannedExperiment] = []

    add_grid(
        plans,
        suite="main_experiment",
        exp_types=["B1", "B2"],
        datasets=settings.paper_datasets,
        imputators=settings.paper_imputators,
        predictors=settings.paper_predictors,
        missing_rates=[main_missing_rate],
        seeds=seeds,
        settings=settings,
        lookback_len=lookback_len,
        tip_k=settings.default_tip_k,
        tip_epsilon=settings.default_tip_epsilon,
    )

    if args.quick:
        projection_datasets = ["Supply_Chain_Customer_D", "azure2019_D_5T"]
        projection_imputators = ["SAITS", "DeepMVI"]
        projection_predictors = ["TimesFM25", "Sundial"]
    else:
        projection_datasets = settings.paper_datasets
        projection_imputators = settings.paper_imputators
        projection_predictors = settings.paper_predictors

    add_grid(
        plans,
        suite="projection_ablation",
        exp_types=["B1", "B2", "B4", "B5", "B6"],
        datasets=projection_datasets,
        imputators=projection_imputators,
        predictors=projection_predictors,
        missing_rates=[main_missing_rate],
        seeds=seeds,
        settings=settings,
        lookback_len=lookback_len,
        tip_k=settings.default_tip_k,
        tip_epsilon=settings.default_tip_epsilon,
    )

    representative_configs = [
        ("Supply_Chain_Customer_D", "SAITS", "TimesFM25"),
        ("azure2019_D_5T", "DeepMVI", "Sundial"),
    ]
    add_paired_configs(
        plans,
        suite="k_sweep",
        exp_type="B2",
        configs=representative_configs,
        values=k_values,
        value_kind="k",
        seeds=seeds,
        settings=settings,
        missing_rate=main_missing_rate,
        lookback_len=lookback_len,
    )
    add_paired_configs(
        plans,
        suite="epsilon_sweep",
        exp_type="B2",
        configs=representative_configs,
        values=epsilon_values,
        value_kind="epsilon",
        seeds=seeds,
        settings=settings,
        missing_rate=main_missing_rate,
        lookback_len=lookback_len,
    )

    add_grid(
        plans,
        suite="missing_rate_robustness",
        exp_types=["B1", "B2", "B3"],
        datasets=["Supply_Chain_Customer_D", "azure2019_D_5T"],
        imputators=["SAITS", "DeepMVI"],
        predictors=settings.paper_predictors,
        missing_rates=robustness_missing_rates,
        seeds=seeds,
        settings=settings,
        lookback_len=lookback_len,
        tip_k=settings.default_tip_k,
        tip_epsilon=settings.default_tip_epsilon,
    )

    compute_suffix = f"costprobe_e{args.compute_max_epoch}"
    for exp_type, tip_k in [("B1", "0"), ("B2", "20"), ("B2", "10"), ("B2", "5"), ("B6", "20")]:
        plans.append(
            PlannedExperiment(
                suite="compute_cost_probe",
                exp_type=exp_type,
                dataset="Supply_Chain_Customer_D",
                imputator="SAITS",
                predictor="TimesFM25",
                missing_rate=main_missing_rate,
                seed="42",
                lookback_len=lookback_len,
                block_size=settings.default_block_size,
                tip_k=tip_k,
                tip_epsilon=settings.default_tip_epsilon,
                tip_pretrain_epochs=settings.default_pretrain_epochs,
                exp_name_suffix=compute_suffix,
            )
        )

    return plans


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


def parse_log(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
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
        **{field: None for field in METRIC_FIELDS},
    }

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    row["completed"] = any("Experiment completed successfully!" in ln for ln in lines)

    section: str | None = None
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
            elif section == "training" and key in {
                "tip_k",
                "tip_epsilon",
                "tip_pretrain_epochs",
            }:
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

    if row["exp_type"] is None:
        row["exp_type"] = str(row["exp_name"]).split("_", 1)[0]

    return row


def parse_tip_stats(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def is_experiment_log(path: Path) -> bool:
    return path.suffix == ".log" and not path.name.endswith(".runner.log")


def discover_logs() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted(path for path in LOG_DIR.glob("*.log") if is_experiment_log(path))


def read_compute_timing() -> dict[tuple[str, str], dict[str, str]]:
    timing_path = RESULT_DIR / "compute_cost_probe.csv"
    if not timing_path.exists():
        return {}
    with timing_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (row.get("strategy", ""), row.get("k", "")): row
            for row in reader
            if row.get("strategy") and row.get("k")
        }


def find_log_for_plan(
    plan: PlannedExperiment,
    log_by_stem: dict[str, Path],
) -> Path | None:
    exact = log_by_stem.get(plan.expected_exp_name)
    if exact:
        return exact

    if plan.suite != "compute_cost_probe" or "_costprobe_e" not in plan.expected_exp_name:
        return None

    prefix = plan.expected_exp_name.split("_costprobe_e", 1)[0] + "_costprobe_e"
    candidates = [path for stem, path in log_by_stem.items() if stem.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def planned_row(
    plan: PlannedExperiment,
    compute_timing: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    timing = compute_timing.get((plan.exp_type, plan.tip_k), {})
    return {
        "suite": plan.suite,
        "suite_order": plan.suite_order,
        "status": "missing_log",
        "completed": False,
        "expected_exp_name": plan.expected_exp_name,
        "exp_name": plan.expected_exp_name,
        "log_file": str((LOG_DIR / f"{plan.expected_exp_name}.log").relative_to(ROOT)),
        "exp_type": plan.exp_type,
        "dataset": plan.dataset,
        "missing_rate": plan.missing_rate,
        "artificial_missing_rate": plan.missing_rate,
        "lookback_len": plan.lookback_len,
        "block_size": plan.block_size,
        "imputator": plan.imputator,
        "predictor": plan.predictor,
        "seed": plan.seed,
        "tip_k": plan.tip_k,
        "tip_epsilon": plan.tip_epsilon,
        "tip_pretrain_epochs": plan.tip_pretrain_epochs,
        **{field: None for field in METRIC_FIELDS},
        "elapsed_seconds": timing.get("elapsed_seconds"),
        "exit_code": timing.get("exit_code"),
    }


def collect_planned_rows(
    plans: list[PlannedExperiment],
    log_by_stem: dict[str, Path],
    compute_timing: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], set[Path]]:
    rows: list[dict[str, Any]] = []
    used_logs: set[Path] = set()

    for plan in plans:
        row = planned_row(plan, compute_timing)
        log_path = find_log_for_plan(plan, log_by_stem)
        if not log_path:
            rows.append(row)
            continue

        used_logs.add(log_path)
        parsed = parse_log(log_path)
        row["status"] = "completed" if parsed.get("completed") else "incomplete"
        row["completed"] = bool(parsed.get("completed"))
        row["log_file"] = parsed.get("log_file") or row["log_file"]
        row["exp_name"] = parsed.get("exp_name") or row["exp_name"]

        for field in CONFIG_FIELDS + METRIC_FIELDS:
            value = parsed.get(field)
            if value not in {None, ""}:
                row[field] = value
        rows.append(row)

    return rows, used_logs


def collect_unplanned_rows(log_paths: list[Path], used_logs: set[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in log_paths:
        if path in used_logs:
            continue
        parsed = parse_log(path)
        row = {
            field: None
            for field in RESULT_FIELDS
        }
        row.update(parsed)
        row["suite"] = "unplanned"
        row["suite_order"] = SUITE_ORDER["unplanned"]
        row["status"] = "completed" if parsed.get("completed") else "incomplete"
        row["expected_exp_name"] = parsed.get("exp_name")
        row["elapsed_seconds"] = None
        row["exit_code"] = None
        rows.append(row)
    return rows


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def baseline_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("missing_rate"),
        row.get("lookback_len"),
        row.get("imputator"),
        row.get("predictor"),
        row.get("seed"),
    )


def stat_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("missing_rate"),
        row.get("lookback_len"),
        row.get("imputator"),
        row.get("predictor"),
        row.get("seed"),
        row.get("tip_k"),
        row.get("tip_epsilon"),
    )


def dedupe_for_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("exp_name") or row.get("expected_exp_name"))
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        current_score = STATUS_SCORE.get(str(current.get("status")), -1)
        new_score = STATUS_SCORE.get(str(row.get("status")), -1)
        if new_score > current_score:
            selected[key] = row
    return list(selected.values())


def collect_relative(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("exp_type") == "B1" and isinstance(row.get("pred_mae"), float):
            baselines[baseline_key(row)] = row

    rel_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("exp_type") == "B1":
            continue
        base = baselines.get(baseline_key(row))
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
                "suite": row.get("suite"),
                "exp_type": row.get("exp_type"),
                "dataset": row.get("dataset"),
                "missing_rate": row.get("missing_rate"),
                "lookback_len": row.get("lookback_len"),
                "imputator": row.get("imputator"),
                "predictor": row.get("predictor"),
                "seed": row.get("seed"),
                "tip_k": row.get("tip_k"),
                "tip_epsilon": row.get("tip_epsilon"),
                "pred_mae": pred_mae,
                "b1_pred_mae": base_pred_mae,
                "pred_mae_ratio": pred_mae / base_pred_mae if base_pred_mae else math.nan,
                "pred_mse": pred_mse,
                "b1_pred_mse": base_pred_mse,
                "pred_mse_ratio": (
                    pred_mse / base_pred_mse
                    if isinstance(pred_mse, float)
                    and isinstance(base_pred_mse, float)
                    and base_pred_mse
                    else math.nan
                ),
            }
        )
    return rel_rows


def collect_summary(rel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rel_rows:
        ratio = row.get("pred_mae_ratio")
        if isinstance(ratio, float) and not math.isnan(ratio):
            grouped[(row.get("suite"), row.get("dataset"), row.get("exp_type"))].append(ratio)

    out: list[dict[str, Any]] = []
    for (suite, dataset, exp_type), values in sorted(grouped.items()):
        out.append(
            {
                "suite": suite,
                "dataset": dataset,
                "exp_type": exp_type,
                "count": len(values),
                "mean_pred_mae_ratio": statistics.fmean(values),
                "median_pred_mae_ratio": statistics.median(values),
                "std_pred_mae_ratio": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "win_rate_vs_b1": sum(v < 1.0 for v in values) / len(values),
            }
        )
    return out


def stat_tests(rows: list[dict[str, Any]]) -> str:
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

    by_key_type: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        pred_mae = row.get("pred_mae")
        exp_type = row.get("exp_type")
        if isinstance(pred_mae, float) and isinstance(exp_type, str):
            by_key_type[stat_key(row)][exp_type] = pred_mae

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
            + ", ".join(f"{key}={value:.3f}" for key, value in avg_ranks.items())
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


def count_by_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"completed": 0, "incomplete": 0, "missing_log": 0}
    for row in rows:
        status = str(row.get("status"))
        if status in counts:
            counts[status] += 1
    return counts


def main() -> None:
    settings = load_common_settings()
    args = parse_args(settings)
    RESULT_DIR.mkdir(exist_ok=True)

    plans = build_plans(args, settings)
    log_paths = discover_logs()
    log_by_stem = {path.stem: path for path in log_paths}
    compute_timing = read_compute_timing()

    planned_rows, used_logs = collect_planned_rows(plans, log_by_stem, compute_timing)
    unplanned_rows = collect_unplanned_rows(log_paths, used_logs)
    all_rows = planned_rows + unplanned_rows

    write_csv(RESULT_DIR / "experiment_results.csv", all_rows, RESULT_FIELDS)

    for suite, filename in SUITE_OUTPUTS.items():
        suite_rows = [row for row in planned_rows if row.get("suite") == suite]
        write_csv(RESULT_DIR / filename, suite_rows, RESULT_FIELDS)

    missing_rows = [row for row in planned_rows if row.get("status") != "completed"]
    write_csv(RESULT_DIR / "missing_experiments.csv", missing_rows, RESULT_FIELDS)
    write_csv(RESULT_DIR / "unplanned_logs.csv", unplanned_rows, RESULT_FIELDS)

    analysis_rows = dedupe_for_analysis(all_rows)
    rel_rows = collect_relative(all_rows)
    rel_fields = [
        "suite",
        "exp_type",
        "dataset",
        "missing_rate",
        "lookback_len",
        "imputator",
        "predictor",
        "seed",
        "tip_k",
        "tip_epsilon",
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
        "suite",
        "dataset",
        "exp_type",
        "count",
        "mean_pred_mae_ratio",
        "median_pred_mae_ratio",
        "std_pred_mae_ratio",
        "win_rate_vs_b1",
    ]
    write_csv(RESULT_DIR / "summary_by_dataset.csv", summary_rows, summary_fields)

    stats_text = stat_tests(analysis_rows)
    (RESULT_DIR / "stat_tests.txt").write_text(stats_text, encoding="utf-8")

    tip_rows = [item for path in sorted(used_logs) for item in parse_tip_stats(path)]
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

    print(f"Wrote {len(all_rows)} rows to {RESULT_DIR / 'experiment_results.csv'}")
    for suite in SUITE_OUTPUTS:
        suite_rows = [row for row in planned_rows if row.get("suite") == suite]
        counts = count_by_status(suite_rows)
        print(
            f"{suite}: total={len(suite_rows)}, completed={counts['completed']}, "
            f"incomplete={counts['incomplete']}, missing_log={counts['missing_log']}"
        )
    print(f"Wrote {len(missing_rows)} placeholder rows to {RESULT_DIR / 'missing_experiments.csv'}")
    print(f"Wrote {len(unplanned_rows)} unplanned log rows to {RESULT_DIR / 'unplanned_logs.csv'}")
    print(f"Wrote {len(rel_rows)} rows to {RESULT_DIR / 'relative_to_b1.csv'}")
    print(f"Wrote summary to {RESULT_DIR / 'summary_by_dataset.csv'}")
    print(f"Wrote statistical tests to {RESULT_DIR / 'stat_tests.txt'}")
    print(f"Wrote {len(tip_rows)} TIP stat rows to {RESULT_DIR / 'tip_epoch_stats.csv'}")


if __name__ == "__main__":
    main()
