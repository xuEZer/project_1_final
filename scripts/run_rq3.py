#!/usr/bin/env python3
"""Run the RQ3 counterfactual component analysis on a shared B1 trajectory.

One training process applies the raw B1 AdamW proposal at every step. Before
it is committed, B2--B5 are evaluated sequentially from the same parameter
snapshot and batch. Scheduling, completion detection, and failure handling
are shared with ``run_main.py``.

Examples:
  uv run python scripts/run_rq3.py --dry-run
  uv run python scripts/run_rq3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_main import main


RQ3_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "rq3_exp.json"


if __name__ == "__main__":
    # Keep run_main.py as the sole implementation of batch scheduling.  The
    # explicit insertion also leaves all its normal CLI filters available.
    if "--config" not in sys.argv[1:]:
        sys.argv[1:1] = ["--config", str(RQ3_CONFIG)]
    raise SystemExit(main())
