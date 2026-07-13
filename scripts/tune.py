#!/usr/bin/env python3
"""Hyperparameter search for B1 imputer training.

Grid search over lr × batch_size, selecting the best configuration
by validation ``imp_loss``.  Experiment names describe the training
configuration itself, so the same run can be reused across analyses.

Usage:
  python scripts/tune.py              # serial (default)
  python scripts/tune.py -j 4         # 4 parallel workers
"""

from __future__ import annotations

import argparse
import json
import re
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
SEARCH_GRID = ROOT / "configs" / "search_grid.json"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "cache" / "ckpt"
BEST_CFG_DIR = ROOT / "configs" / "best_cfg"

SEARCH_PREDICTOR = "Chronos"

_print_lock = threading.Lock()


def log(msg: str = "", end: str = "\n") -> None:
    with _print_lock:
        print(msg, end=end, flush=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_completed(name: str) -> bool:
    """Check if experiment already ran successfully."""
    log_path = LOG_DIR / f"{name}.log"
    if not log_path.exists():
        return False
    text = log_path.read_text()
    return "Experiment completed successfully!" in text


def parse_result(name: str, lr: float, batch_size: int) -> dict:
    """Parse the result from a completed experiment log."""
    log_path = LOG_DIR / f"{name}.log"
    if not log_path.exists():
        return {"name": name, "lr": lr, "batch_size": batch_size, "val_imp_loss": float("inf")}

    text = log_path.read_text()
    m = re.search(r"Restored best val checkpoint: epoch=(\d+), imp_loss=([\d.]+)", text)
    if m:
        return {
            "name": name,
            "lr": lr,
            "batch_size": batch_size,
            "epoch": int(m.group(1)),
            "val_imp_loss": float(m.group(2)),
        }
    return {"name": name, "lr": lr, "batch_size": batch_size, "val_imp_loss": float("inf")}


def run_one(dataset: str, imputer: str, lr: float, batch_size: int) -> dict:
    """Run a single B1 experiment and return parsed results."""
    overrides = [
        "--experiment.type", "B1",
        "--data.dataset", dataset,
        "--imputer.model", imputer,
        "--predictor.model", SEARCH_PREDICTOR,
        "--training.lr", str(lr),
        "--loader.batch_size", str(batch_size),
    ]
    name = resolve_experiment_name(overrides)

    # Skip if already completed
    if is_completed(name):
        return parse_result(name, lr, batch_size)

    cmd = [sys.executable, str(ROOT / "main.py"), *overrides]
    ret = subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
    if ret.returncode != 0:
        return {"name": name, "lr": lr, "batch_size": batch_size, "val_imp_loss": float("inf")}

    return parse_result(name, lr, batch_size)


def run_serial(groups: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], list[dict]]:
    """Run tasks sequentially, with a header per (dataset, imputer) group."""
    results: dict[tuple[str, str], list[dict]] = {}
    for key, group_tasks in groups.items():
        dataset, imputer = key
        log(f"\n{'='*60}")
        log(f"  Search: dataset={dataset}  imputer={imputer}")
        log(f"{'='*60}")
        results[key] = []
        for task in group_tasks:
            log(f"  lr={task['lr']}  batch_size={task['batch_size']} ...", end=" ")
            r = run_one(dataset, imputer, task["lr"], task["batch_size"])
            results[key].append(r)
            if r["val_imp_loss"] != float("inf"):
                log(f"val_imp_loss={r['val_imp_loss']:.6f}")
            else:
                log("FAILED")
    return results


def run_parallel(tasks: list[dict], workers: int) -> dict[tuple[str, str], list[dict]]:
    """Run tasks in parallel, grouped by (dataset, imputer)."""
    results: dict[tuple[str, str], list[dict]] = {}

    def worker(task: dict) -> dict:
        log(f"  {task['dataset']}/{task['imputer']}  lr={task['lr']}  bs={task['batch_size']}  ...", end=" ")
        r = run_one(task["dataset"], task["imputer"], task["lr"], task["batch_size"])
        if r["val_imp_loss"] != float("inf"):
            log(f"val_imp_loss={r['val_imp_loss']:.6f}")
        else:
            log("FAILED")
        return r

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                key = (task["dataset"], task["imputer"])
                r = future.result()
                if key not in results:
                    results[key] = []
                results[key].append(r)
    except KeyboardInterrupt:
        log("\n[ABORT] Interrupted; cancelling pending tasks...")
        for f in futures:
            f.cancel()
        pool.shutdown(wait=False)
        raise

    return results


def save_best_config(dataset: str, imputer: str, best: dict) -> Path:
    """Write best config .ini file and return its path."""
    ckpt_path = CKPT_DIR / f"{best['name']}.pt"
    best_cfg = BEST_CFG_DIR / f"best_{dataset}_{imputer}.ini"
    best_cfg.write_text(
        f"[training]\n"
        f"lr = {best['lr']}\n"
        f"\n"
        f"[loader]\n"
        f"batch_size = {best['batch_size']}\n"
        f"\n"
        f"[paths]\n"
        f"ckpt = {ckpt_path}\n"
    )
    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 hyperparameter grid search")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    main_exp = load_json(MAIN_EXP)
    grid = load_json(SEARCH_GRID)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CFG_DIR.mkdir(parents=True, exist_ok=True)

    # Build grouped tasks: (dataset, imputer) → list of {lr, batch_size}
    groups: dict[tuple[str, str], list[dict]] = {}
    for dataset, imputer in product(main_exp["datasets"], main_exp["imputers"]):
        key = (dataset, imputer)
        groups[key] = []
        for lr, batch_size in product(grid["lr"], grid["batch_size"]):
            groups[key].append({"lr": lr, "batch_size": batch_size})

    total = sum(len(v) for v in groups.values())
    log(f"{'='*60}")
    log(f"  {len(groups)} dataset×imputer pairs, {total} runs, {args.workers} worker(s)")
    log(f"{'='*60}")

    # Run
    if args.workers <= 1:
        results = run_serial(groups)
    else:
        # Flatten for parallel execution
        flat_tasks = [
            {"dataset": ds, "imputer": imp, "lr": t["lr"], "batch_size": t["batch_size"]}
            for (ds, imp), ts in groups.items() for t in ts
        ]
        results = run_parallel(flat_tasks, args.workers)

    # Process results: select best per (dataset, imputer)
    log(f"\n{'='*60}")
    log("  Selecting best configs")
    log(f"{'='*60}")

    total_ok = 0
    total_fail = 0
    for (dataset, imputer), runs in sorted(results.items()):
        valid = [r for r in runs if r["val_imp_loss"] != float("inf")]
        n_fail = len(runs) - len(valid)
        total_ok += len(valid)
        total_fail += n_fail

        if valid:
            best = min(valid, key=lambda r: r["val_imp_loss"])
            best_cfg = save_best_config(dataset, imputer, best)
            log(f"  {dataset}/{imputer}: best lr={best['lr']} bs={best['batch_size']} "
                f"val_imp_loss={best['val_imp_loss']:.6f}  →  {best_cfg}")
        else:
            log(f"  {dataset}/{imputer}: ALL {len(runs)} RUNS FAILED")

    log(f"\n{'='*60}")
    log(f"  Done: {total_ok} OK, {total_fail} FAILED")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
