#!/usr/bin/env python3
"""Run the fixed-protocol real-task experiment batch.

The experiment matrix and every shared hyperparameter are defined in
``configs/main_exp.json``. Completed runs are detected from their log files and
skipped, so an interrupted batch can be resumed with the same command.

Examples:
  uv run python scripts/run_main.py --dry-run
  uv run python scripts/run_main.py
  uv run python scripts/run_main.py --workers 2
  uv run python scripts/run_main.py --stage B2 --dataset CPHL_30T
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import config_from_overrides, normalize_config_value, selected_config_payload

DEFAULT_CONFIG_PATH = ROOT / "configs" / "main_exp.json"
LOG_DIR = ROOT / "logs"
_PRINT_LOCK = threading.Lock()


def emit(message: str = "") -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"exp_types", "datasets", "imputers", "predictors", "fixed"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing keys in {config_path}: {sorted(missing)}")
    if any(name.startswith("ETT") for name in config["datasets"]):
        raise ValueError("ETT datasets are excluded from the real-task batch.")
    return config


def fixed_overrides(fixed: dict) -> list[str]:
    mapping = (
        ("experiment.seed", "seed"),
        ("training.lr", "learning_rate"),
        ("loader.batch_size", "batch_size"),
        ("training.max_epochs", "max_epochs"),
        ("training.patience", "patience"),
        ("data.missing_rate", "missing_rate"),
        ("data.artificial_missing_rate", "artificial_missing_rate"),
        ("data.block_size", "block_size"),
        ("data.lookback", "lookback"),
        ("data.pred_len", "prediction_length"),
        ("tip.subspace_rank", "subspace_rank"),
        ("tip.epsilon", "epsilon"),
        ("tip.confidence_delta", "confidence_delta"),
    )
    overrides: list[str] = []
    for option, key in mapping:
        if key not in fixed:
            raise ValueError(f"Missing fixed hyperparameter: {key}")
        overrides.extend((f"--{option}", str(fixed[key])))
    return overrides


def task_overrides(
    exp_type: str,
    dataset: str,
    imputer: str,
    predictor: str,
    fixed: dict,
) -> list[str]:
    return [
        "--experiment.type",
        exp_type,
        "--data.dataset",
        dataset,
        "--imputer.model",
        imputer,
        "--predictor.model",
        predictor,
        *fixed_overrides(fixed),
    ]


def logged_config(path: Path) -> dict[str, str]:
    """Recover normalized ``section.key`` values from one experiment log."""
    values: dict[str, str] = {}
    section: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        message = line.split(" - INFO - ", 1)[-1]
        if message.startswith("[") and message.endswith("]"):
            section = message[1:-1]
            continue
        if section is None or not message.startswith("  ") or " = " not in message:
            continue
        key, value = message.strip().split("=", 1)
        values[f"{section}.{key.strip()}"] = normalize_config_value(value)
    return values


def completed(task: dict) -> bool:
    """Check exact or legacy logs for the same scientific configuration."""
    exact_path = LOG_DIR / f"{task['name']}.log"
    candidates = [exact_path]
    if "__" in task["name"]:
        prefix = task["name"].rsplit("__", 1)[0]
        candidates.extend(
            path for path in LOG_DIR.glob(f"{prefix}__*.log") if path != exact_path
        )
    for path in candidates:
        if not path.exists():
            continue
        content = path.read_text(errors="replace")
        if "Experiment completed successfully!" not in content:
            continue
        if path == exact_path:
            return True
        legacy_values = logged_config(path)
        if all(legacy_values.get(key) == value for key, value in task["payload"].items()):
            return True
    return False


def build_tasks(config: dict, args: argparse.Namespace) -> list[dict]:
    stages = [args.stage] if args.stage else config["exp_types"]
    datasets = [args.dataset] if args.dataset else config["datasets"]
    imputers = [args.imputer] if args.imputer else config["imputers"]
    predictors = [args.predictor] if args.predictor else config["predictors"]

    for selected, available, label in (
        (stages, config["exp_types"], "stage"),
        (datasets, config["datasets"], "dataset"),
        (imputers, config["imputers"], "imputer"),
        (predictors, config["predictors"], "predictor"),
    ):
        unknown = set(selected).difference(available)
        if unknown:
            raise ValueError(f"Unknown {label}: {sorted(unknown)}")

    tasks = []
    # A model combination is the scheduling unit. Its stages stay adjacent,
    # e.g. B1 -> B2 for the main experiment, rather than running all B1 first.
    for dataset, imputer, predictor in product(datasets, imputers, predictors):
        group = f"{dataset}/{imputer}/{predictor}"
        for stage in stages:
            overrides = task_overrides(
                stage, dataset, imputer, predictor, config["fixed"]
            )
            final_config = config_from_overrides(overrides)
            name = final_config.get("experiment", "name")
            tasks.append(
                {
                    "label": f"{stage} {group}",
                    "group": group,
                    "name": name,
                    "payload": selected_config_payload(final_config),
                    "overrides": overrides,
                }
            )
    return tasks


def run_one(task: dict) -> tuple[dict, str]:
    if completed(task):
        return task, "cached"
    command = [sys.executable, str(ROOT / "main.py"), *task["overrides"]]
    result = subprocess.run(command, cwd=ROOT, check=False)
    # ``main.py`` writes the completion marker only after training, checkpoint
    # saving, and all split evaluations have finished.  Some predictor
    # runtimes can still return a non-zero code during interpreter/CUDA
    # teardown.  Treat the durable marker as authoritative so a successfully
    # completed B1 does not prevent its paired B2 from being scheduled.
    if completed(task):
        if result.returncode != 0:
            emit(
                f"[WARN  ] {task['label']} completed, but its process exited "
                f"with code {result.returncode}; continuing from the log "
                "completion marker."
            )
        return task, "ok"
    return task, "failed"


def run_group(tasks: list[dict]) -> list[tuple[dict, str]]:
    """Run all stages for one model combination in configured order."""
    results = []
    for task in tasks:
        task_result = run_one(task)
        results.append(task_result)
        if task_result[1] == "failed":
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real-task experiments with one fixed hyperparameter set."
    )
    parser.add_argument("--workers", "-j", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--dataset")
    parser.add_argument("--imputer")
    parser.add_argument("--predictor")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config = load_config(args.config.expanduser().resolve())
    tasks = build_tasks(config, args)
    pending = [task for task in tasks if not completed(task)]
    emit(
        f"Fixed protocol: {len(tasks)} tasks; "
        f"{len(tasks) - len(pending)} completed; {len(pending)} pending."
    )
    emit(json.dumps(config["fixed"], ensure_ascii=False, sort_keys=True))

    if args.dry_run:
        for index, task in enumerate(tasks, start=1):
            state = "DONE" if completed(task) else "TODO"
            emit(f"[{index:03d}/{len(tasks):03d}] {state} {task['label']}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "cached": len(tasks) - len(pending), "failed": 0}
    groups: dict[str, list[dict]] = {}
    for task in pending:
        groups.setdefault(task["group"], []).append(task)

    if args.workers == 1:
        group_results = map(run_group, groups.values())
        for results in group_results:
            for task, state in results:
                counts[state] += 1
                emit(f"[{state.upper():6}] {task['label']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_group, group_tasks)
                for group_tasks in groups.values()
            ]
            for future in as_completed(futures):
                for task, state in future.result():
                    counts[state] += 1
                    emit(f"[{state.upper():6}] {task['label']}")

    emit(
        f"Finished: {counts['ok']} new, {counts['cached']} cached, "
        f"{counts['failed']} failed."
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
