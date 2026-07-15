#!/usr/bin/env python3
"""Grid-search TIP hyperparameters using validation prediction loss.

Each run still uses validation imputation loss for early stopping and checkpoint
selection. Hyperparameter configurations are ranked by prediction MSE evaluated
on that restored checkpoint. The search is fixed to
Supply_Chain_Customer_D + SAITS + Toto2.

Usage:
  uv run python scripts/tune_tip.py          # inspect/re-rank completed runs
  uv run python scripts/tune_tip.py --run    # run missing combinations serially
  uv run python scripts/tune_tip.py --run -j 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import resolve_experiment_name

DATASET = "Supply_Chain_Customer_D"
IMPUTER = "SAITS"
PREDICTOR = "Toto2"
GRID_PATH = ROOT / "configs" / "tip_search_grid.json"
BEST_CFG_PATH = ROOT / "configs" / "best_cfg" / f"best_{DATASET}_{IMPUTER}.ini"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "cache" / "ckpt"
RESULT_PATH = ROOT / "results" / f"tip_grid_{DATASET}_{IMPUTER}_{PREDICTOR}.json"


def read_best_lr_bs() -> tuple[str, str]:
    section = ""
    values: dict[str, str] = {}
    for raw in BEST_CFG_PATH.read_text().splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line:
            key, value = line.split("=", 1)
            values[f"{section}.{key.strip()}"] = value.strip()
    return values["training.lr"], values["loader.batch_size"]


def overrides_for(k: int, epsilon: float, delta: float, lr: str, bs: str) -> list[str]:
    return [
        "--experiment.type", "B2",
        "--data.dataset", DATASET,
        "--imputer.model", IMPUTER,
        "--predictor.model", PREDICTOR,
        "--training.lr", lr,
        "--loader.batch_size", bs,
        "--tip.k", str(k),
        "--tip.epsilon", str(epsilon),
        "--tip.confidence_delta", str(delta),
    ]


def parse_run(task: dict) -> dict:
    path = LOG_DIR / f"{task['name']}.log"
    result = dict(task)
    result["completed"] = False
    result["training_completed"] = False
    result["best_epoch"] = None
    result["val_imp_loss"] = None
    result["val_pred_mse"] = None
    if not path.exists():
        return result
    text = path.read_text()
    result["training_completed"] = "Experiment completed successfully!" in text
    checkpoint = re.search(
        r"Restored best val checkpoint: epoch=(\d+), imp_loss=([\d.eE+-]+)", text
    )
    if checkpoint is not None:
        result["best_epoch"] = int(checkpoint.group(1))
        result["val_imp_loss"] = float(checkpoint.group(2))
    val_summary = re.search(
        r"Evaluation Results Summary: split=val(?P<body>.*?)"
        r"Evaluation completed on val split\.",
        text,
        flags=re.DOTALL,
    )
    if val_summary is not None:
        mse = re.search(r"\bMSE\b\s+[-+\d.eE]+\s+([-+\d.eE]+)",
                        val_summary.group("body"))
        if mse is not None:
            result["val_pred_mse"] = float(mse.group(1))
    result["completed"] = result["training_completed"] and result["val_pred_mse"] is not None
    return result


def run_task(task: dict) -> dict:
    existing = parse_run(task)
    if existing["completed"]:
        return existing

    log_path = LOG_DIR / f"{task['name']}.log"
    backup_path = LOG_DIR / f"{task['name']}.pre_val_eval.log"
    overrides = list(task["overrides"])
    evaluating_existing = existing["training_completed"]
    if evaluating_existing:
        ckpt_path = CKPT_DIR / f"{task['name']}.pt"
        if not ckpt_path.exists():
            existing["error"] = f"checkpoint missing: {ckpt_path}"
            return existing
        suffix = 1
        while backup_path.exists():
            backup_path = LOG_DIR / f"{task['name']}.pre_val_eval.{suffix}.log"
            suffix += 1
        log_path.replace(backup_path)
        overrides.extend(["--experiment.load_ckpt", str(ckpt_path)])

    process = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *overrides], cwd=ROOT
    )
    result = parse_run(task)
    result["returncode"] = process.returncode
    if evaluating_existing:
        result["best_epoch"] = existing["best_epoch"]
        result["val_imp_loss"] = existing["val_imp_loss"]
    if process.returncode != 0 and evaluating_existing:
        if log_path.exists():
            log_path.unlink()
        backup_path.replace(log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run missing combinations")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Parallel workers (default: 1; safest for one GPU)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    grid = json.loads(GRID_PATH.read_text())
    lr, bs = read_best_lr_bs()
    tasks = []
    for k, epsilon, delta in product(
        grid["k"], grid["epsilon"], grid["confidence_delta"]
    ):
        overrides = overrides_for(k, epsilon, delta, lr, bs)
        tasks.append({
            "k": k,
            "epsilon": epsilon,
            "confidence_delta": delta,
            "lr": lr,
            "batch_size": bs,
            "name": resolve_experiment_name(overrides),
            "overrides": overrides,
        })

    completed = sum(parse_run(task)["completed"] for task in tasks)
    print(f"Grid: {len(tasks)} combinations; {completed} completed; "
          f"{len(tasks) - completed} missing")
    print(f"Fixed setting: {DATASET}/{IMPUTER}/{PREDICTOR}, lr={lr}, bs={bs}")

    if args.run and args.workers == 1:
        results = []
        for index, task in enumerate(tasks, 1):
            print(f"[{index:02d}/{len(tasks)}] k={task['k']} eps={task['epsilon']} "
                  f"delta={task['confidence_delta']}", flush=True)
            results.append(run_task(task))
    elif args.run:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, task): task for task in tasks}
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"done k={result['k']} eps={result['epsilon']} "
                      f"delta={result['confidence_delta']} "
                      f"val_pred_mse={result['val_pred_mse']}", flush=True)
    else:
        results = [parse_run(task) for task in tasks]

    ranked = sorted(
        results,
        key=lambda item: item["val_pred_mse"]
        if item["val_pred_mse"] is not None else math.inf,
    )
    serializable = [{k: v for k, v in item.items() if k != "overrides"}
                    for item in ranked]
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(serializable, indent=2) + "\n")

    valid = [item for item in ranked if item["val_pred_mse"] is not None]
    print(f"\nCompleted results: {len(valid)}/{len(tasks)}")
    for rank, item in enumerate(valid[:10], 1):
        print(f"{rank:2d}. val_pred_mse={item['val_pred_mse']:.6f} "
              f"val_imp_loss={item['val_imp_loss']} k={item['k']} "
              f"eps={item['epsilon']} delta={item['confidence_delta']} "
              f"epoch={item['best_epoch']}")
    if valid:
        best = valid[0]
        print("\nBEST: "
              f"k={best['k']}, epsilon={best['epsilon']}, "
              f"confidence_delta={best['confidence_delta']}, "
              f"val_pred_mse={best['val_pred_mse']:.6f}, "
              f"val_imp_loss={best['val_imp_loss']}")
    if not args.run and len(valid) < len(tasks):
        print("\nRun missing combinations with:")
        print("  uv run python scripts/tune_tip.py --run")
    print(f"Ranking written to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
