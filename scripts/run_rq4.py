#!/usr/bin/env python3
"""Run all RQ4 sensitivity candidates on one realised B1 trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

from run_main import main


RQ4_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "rq4_exp.json"


if __name__ == "__main__":
    if "--config" not in sys.argv[1:]:
        sys.argv[1:1] = ["--config", str(RQ4_CONFIG)]
    raise SystemExit(main())
