#!/usr/bin/env python3
"""Run the impute-then-forecast matrix through expman.

Replaces scripts/run_main.py: the matrix comes from configs/experiment.yaml,
scheduling and resume come from expman.Batch, and each run lives in its own
output directory with per-stage snapshots.

  python run_expman.py                            # the full 160-run matrix
  python run_expman.py --config my.yaml           # a different matrix
  python run_expman.py --resume --output-dir runs/<id>   # continue
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from expman import Batch  # noqa: E402

from src.pipeline import build_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "experiment.yaml"),
        help="experiment YAML with !choice fields",
    )
    parser.add_argument("--output-dir", default=None, help="batch output directory")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an existing --output-dir instead of reading the YAML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline()

    if args.resume:
        if not args.output_dir:
            raise SystemExit("--resume requires --output-dir")
        batch = Batch.resume(pipeline, args.output_dir)
    else:
        batch = Batch(
            pipeline,
            args.config,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
        )

    print(f"output_dir: {batch.output_dir}")
    results = batch.run()

    ok = 0
    for result in results:
        status = getattr(result.status, "value", result.status)
        if status == "succeeded":
            ok += 1
            metrics = result.output.get("test", {}) if result.output else {}
            print(f"[{status:>9}] {result.run_id}  test_pred_mse="
                  f"{metrics.get('pred_mse')}")
        else:
            print(f"[{status:>9}] {result.run_id}")
    print(f"finished: {ok}/{len(results)} succeeded")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
