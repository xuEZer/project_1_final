#!/usr/bin/env python3
"""Run configured B1/B2/B3 experiments.

- B1: loads the selected imputer checkpoint and evaluates each predictor
  directly (no retraining).
- B2: reads the selected lr/batch_size only; TIP starts from random
  initialization and trains independently for each predictor.
- B3: reads the selected lr/batch_size only; starts from random initialization,
  pretrains with imputation loss, then runs TIP fine-tuning.

Usage:
  python scripts/run_main.py           # serial (default)
  python scripts/run_main.py -j 4      # 4 parallel workers
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

from main import resolve_experiment_name

MAIN_EXP = ROOT / "configs" / "main_exp.json"
LOG_DIR = ROOT / "logs"
BEST_CFG_DIR = ROOT / "configs" / "best_cfg"

_print_lock = threading.Lock()


def log(msg: str = "", end: str = "\n") -> None:
    with _print_lock:
        print(msg, end=end, flush=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def read_best_config(dataset: str, imputer: str) -> dict[str, str]:
    """Parse best_{dataset}_{imputer}.ini into a dict."""
    cfg_path = BEST_CFG_DIR / f"best_{dataset}_{imputer}.ini"
    if not cfg_path.exists():
        log(f"  [WARN] Best config not found: {cfg_path}")
        return {}
    cfg: dict[str, str] = {}
    section = ""
    for line in cfg_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line and section:
            k, v = line.split("=", 1)
            cfg[f"{section}.{k.strip()}"] = v.strip()
    return cfg


def is_completed(name: str) -> bool:
    log_path = LOG_DIR / f"{name}.log"
    if not log_path.exists():
        return False
    return "Experiment completed successfully!" in log_path.read_text()


def run_one(exp_type: str, dataset: str, imputer: str,
            predictor: str, lr: str, batch_size: str,
            ckpt: str = "") -> tuple[str, bool, bool]:
    """Run one experiment; return (name, ok, was_cached)."""
    overrides = [
        "--experiment.type", exp_type,
        "--data.dataset", dataset,
        "--imputer.model", imputer,
        "--predictor.model", predictor,
        "--training.lr", lr,
        "--loader.batch_size", batch_size,
    ]
    if ckpt:
        overrides += ["--experiment.load_ckpt", ckpt]

    name = resolve_experiment_name(overrides)
    if is_completed(name):
        return name, True, True

    cmd = [sys.executable, str(ROOT / "main.py"), *overrides]
    subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
    return name, is_completed(name), False


def build_tasks(main_exp: dict, best_cfgs: dict) -> tuple[list[dict], int]:
    """Build flat task list; return (tasks, skip_count)."""
    tasks: list[dict] = []
    skip_count = 0

    for exp_type in main_exp["exp_types"]:
        for (dataset, imputer), cfg in best_cfgs.items():
            lr = cfg.get("training.lr", "1e-3")
            bs = cfg.get("loader.batch_size", "32")
            ckpt = cfg.get("paths.ckpt", "")

            for predictor in main_exp["predictors"]:
                if exp_type == "B1" and not ckpt:
                    log(f"  B1 {dataset}/{imputer}/{predictor} SKIP (no ckpt)")
                    skip_count += 1
                    continue
                tasks.append({
                    "exp_type": exp_type, "dataset": dataset,
                    "imputer": imputer, "predictor": predictor,
                    "lr": lr, "bs": bs,
                    "ckpt": ckpt if exp_type == "B1" else "",
                    "label": f"{exp_type} {dataset}/{imputer}/{predictor}",
                })
    return tasks, skip_count


def run_serial(tasks: list[dict]) -> tuple[int, int, int]:
    ok = fail = cached = 0
    label_width = max(len(t["label"]) for t in tasks) if tasks else 40
    for task in tasks:
        log(f"  {task['label']:<{label_width}}  ...", end=" ")
        name, success, was_cached = run_one(
            task["exp_type"], task["dataset"], task["imputer"],
            task["predictor"], task["lr"], task["bs"],
            ckpt=task["ckpt"],
        )
        if was_cached:
            log("OK (cached)")
            ok += 1
            cached += 1
        elif success:
            log("OK")
            ok += 1
        else:
            log("FAILED")
            fail += 1
    return ok, fail, cached


def run_parallel(tasks: list[dict], workers: int) -> tuple[int, int, int]:
    ok = fail = cached = 0
    failed_labels: list[str] = []

    def worker(task: dict) -> tuple[str, bool, bool]:
        name, success, was_cached = run_one(
            task["exp_type"], task["dataset"], task["imputer"],
            task["predictor"], task["lr"], task["bs"],
            ckpt=task["ckpt"],
        )
        return task["label"], success, was_cached

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, t): t for t in tasks}
            for future in as_completed(futures):
                label, success, was_cached = future.result()
                if was_cached:
                    log(f"  {label}  OK (cached)")
                    ok += 1
                    cached += 1
                elif success:
                    log(f"  {label}  OK")
                    ok += 1
                else:
                    log(f"  {label}  FAILED")
                    fail += 1
                    failed_labels.append(label)
    except KeyboardInterrupt:
        log("\n[ABORT] Interrupted; cancelling pending tasks...")
        for f in futures:
            f.cancel()
        pool.shutdown(wait=False)
        raise

    if failed_labels:
        log(f"\n  Failed tasks ({len(failed_labels)}):")
        for label in failed_labels:
            log(f"    {label}")

    return ok, fail, cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured experiments")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    main_exp = load_json(MAIN_EXP)

    best_cfgs: dict[tuple[str, str], dict[str, str]] = {}
    for dataset, imputer in product(main_exp["datasets"], main_exp["imputers"]):
        cfg = read_best_config(dataset, imputer)
        if cfg:
            best_cfgs[(dataset, imputer)] = cfg
        else:
            log(f"[SKIP] {dataset}/{imputer}: no best config")

    tasks, skip_count = build_tasks(main_exp, best_cfgs)

    log(f"\n{'='*60}")
    log(f"  {len(tasks)} tasks, {skip_count} skipped, {args.workers} worker(s)")
    log(f"{'='*60}\n")

    if args.workers <= 1:
        ok, fail, cached = run_serial(tasks)
    else:
        ok, fail, cached = run_parallel(tasks, args.workers)

    log(f"\n{'='*60}")
    parts = [f"{ok} OK"]
    if cached:
        parts.append(f"{cached} cached")
    parts.append(f"{fail} FAILED")
    if skip_count:
        parts.append(f"{skip_count} SKIP")
    log(f"  Done: {', '.join(parts)}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
