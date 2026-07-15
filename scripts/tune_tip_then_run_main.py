#!/usr/bin/env python3
"""Tune TIP, write the validation-best parameters, then run main experiments.

Usage:
  uv run python scripts/tune_tip_then_run_main.py
  uv run python scripts/tune_tip_then_run_main.py --tune-workers 2 --main-workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE_SCRIPT = ROOT / "scripts" / "tune_tip.py"
MAIN_SCRIPT = ROOT / "scripts" / "run_main.py"
GLOBAL_CONFIG = ROOT / "configs" / "global.ini"
GRID_PATH = ROOT / "configs" / "tip_search_grid.json"
RESULT_PATH = ROOT / "results" / "tip_grid_Supply_Chain_Customer_D_SAITS_Toto2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune-workers", type=int, default=1,
        help="Parallel workers for tune_tip.py (default: 1)",
    )
    parser.add_argument(
        "--main-workers", type=int, default=1,
        help="Parallel workers for run_main.py (default: 1)",
    )
    args = parser.parse_args()
    if args.tune_workers < 1 or args.main_workers < 1:
        parser.error("worker counts must be at least 1")
    return args


def run_checked(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_complete_best_result() -> tuple[dict, int]:
    grid = json.loads(GRID_PATH.read_text())
    expected = math.prod(
        len(grid[key]) for key in ("k", "epsilon", "confidence_delta")
    )
    results = json.loads(RESULT_PATH.read_text())
    complete = [
        row for row in results
        if row.get("completed")
        and row.get("val_pred_mse") is not None
        and math.isfinite(float(row["val_pred_mse"]))
    ]
    combinations = {
        (row["k"], row["epsilon"], row["confidence_delta"])
        for row in complete
    }
    expected_combinations = set(product(
        grid["k"], grid["epsilon"], grid["confidence_delta"]
    ))
    if len(complete) != expected or combinations != expected_combinations:
        missing = expected_combinations - combinations
        raise RuntimeError(
            f"TIP search is incomplete: {len(combinations)}/{expected} unique "
            f"combinations completed; missing={sorted(missing)}"
        )
    return min(complete, key=lambda row: float(row["val_pred_mse"])), expected


def update_tip_section(best: dict) -> None:
    replacements = {
        "k": str(best["k"]),
        "epsilon": str(best["epsilon"]),
        "confidence_delta": str(best["confidence_delta"]),
    }
    lines = GLOBAL_CONFIG.read_text().splitlines(keepends=True)
    section: str | None = None
    updated: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section != "tip" or "=" not in line or stripped.startswith(("#", ";")):
            continue
        key = line.split("=", 1)[0].strip()
        if key in replacements:
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"{key} = {replacements[key]}{newline}"
            updated.add(key)
    missing = replacements.keys() - updated
    if missing:
        raise RuntimeError(f"Missing keys in [tip] section: {sorted(missing)}")

    temporary = GLOBAL_CONFIG.with_suffix(".ini.tmp")
    temporary.write_text("".join(lines))
    temporary.replace(GLOBAL_CONFIG)


def main() -> None:
    args = parse_args()
    tune_command = [sys.executable, str(TUNE_SCRIPT), "--run"]
    if args.tune_workers != 1:
        tune_command.extend(["--workers", str(args.tune_workers)])
    run_checked(tune_command, "1/3 Tune TIP")

    best, total = load_complete_best_result()
    print(
        f"\n[2/3 Select TIP] evaluated={total} | "
        f"k={best['k']} | epsilon={best['epsilon']} | "
        f"confidence_delta={best['confidence_delta']} | "
        f"val_pred_mse={float(best['val_pred_mse']):.6f}",
        flush=True,
    )
    update_tip_section(best)
    print(f"Updated {GLOBAL_CONFIG}", flush=True)

    main_command = [sys.executable, str(MAIN_SCRIPT)]
    if args.main_workers != 1:
        main_command.extend(["--workers", str(args.main_workers)])
    run_checked(main_command, "3/3 Run main experiments")


if __name__ == "__main__":
    main()
