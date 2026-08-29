#!/usr/bin/env python3
"""Run three-seed RQ4 shared-trajectory wall-clock measurements."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_main import load_config, task_overrides

RQ4_CONFIG = ROOT / "configs" / "rq4_timing_exp.json"
DEFAULT_OUTPUT = ROOT / "results" / "rq4_timing"


def completed(log_dir: Path) -> bool:
    return any(
        "Experiment completed successfully!" in path.read_text(
            encoding="utf-8", errors="replace"
        )
        for path in log_dir.glob("RQ4_*.log")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure RQ4 epoch and rank-wise TIP wall-clock time."
    )
    parser.add_argument("--config", type=Path, default=RQ4_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.expanduser().resolve())
    if config["exp_types"] != ["RQ4"]:
        parser.error("the timing config must contain exactly exp_types=[\"RQ4\"]")
    if (
        len(config["datasets"]) != 1
        or len(config["imputers"]) != 1
        or len(config["predictors"]) != 1
    ):
        parser.error("the timing config must select one dataset/imputer/predictor tuple")

    seeds = config.get("seeds", [config["fixed"]["seed"]])
    output_dir = args.output_dir.expanduser().resolve()
    failures = 0
    for seed in seeds:
        fixed = {**config["fixed"], "seed": seed}
        overrides = task_overrides(
            "RQ4",
            config["datasets"][0],
            config["imputers"][0],
            config["predictors"][0],
            fixed,
        )
        log_dir = output_dir / f"seed_{seed}" / "logs"
        command = [
            sys.executable,
            str(ROOT / "main.py"),
            *overrides,
            "--paths.log_dir",
            str(log_dir),
        ]
        state = "completed" if completed(log_dir) else "pending"
        print(f"seed={seed} state={state} log_dir={log_dir}", flush=True)
        if args.dry_run or state == "completed":
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0 or not completed(log_dir):
            failures += 1
            print(f"seed={seed} failed exit_code={result.returncode}", flush=True)

    if failures:
        return 1
    if not args.dry_run:
        print(
            "Timing runs finished. Collect with: "
            f"{sys.executable} scripts/collect_rq4_timing.py --input-dir {output_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
