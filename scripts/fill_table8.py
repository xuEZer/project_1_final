#!/usr/bin/env python3
"""Fill the RQ4 sensitivity table from the completed shared-trajectory log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from collect_rq4 import aggregate, completed_seed_logs
except ModuleNotFoundError:
    from scripts.collect_rq4 import aggregate, completed_seed_logs


ROOT = Path(__file__).resolve().parent.parent
TABLE_LABEL = r"\label{tab:rq4_sensitivity}"


def fmt_pct(value) -> str:
    return r"\ldots" if value == "" else f"{float(value):.2f}"


def fmt_gain(row: dict) -> str:
    mean = row["relative_gain_mean"]
    std = row["relative_gain_std"]
    if mean == "" or std == "":
        return r"$\ldots\pm\ldots$"
    return rf"${float(mean):.4f}\pm{float(std):.4f}$"


def cells(row: dict | None) -> str:
    if row is None:
        return r"\ldots & \ldots & \ldots & \ldots & $\ldots\pm\ldots$"
    return " & ".join([
        fmt_pct(row["prediction_nonincrease_rate"]),
        fmt_pct(row["projection_rate"]),
        fmt_pct(row["false_positive_rate"]),
        fmt_pct(row["intervention_improve_rate"]),
        fmt_gain(row),
    ])


def make_rows(results: list[dict]) -> str:
    lookup = {(row["sweep"], float(row["value"])): row for row in results if row["sweep"] != "B1"}
    lines = []
    for index, rank in enumerate((2, 4, 8, 16)):
        label = r"$r$" if index == 0 else ""
        lines.append(rf"{label} & {rank} & {cells(lookup.get(('rank', float(rank))))} \\")
    lines.append(r"\midrule")
    for index, epsilon in enumerate((1e-5, 1e-4, 1e-3, 1e-2, 1e-1)):
        label = r"$\varepsilon$" if index == 0 else ""
        exponent = int(round(-__import__("math").log10(epsilon)))
        shown = rf"$10^{{-{exponent}}}$"
        lines.append(rf"{label} & {shown} & {cells(lookup.get(('epsilon', epsilon)))} \\")
    lines.append(r"\midrule")
    for index, delta in enumerate((0.01, 0.05, 0.1, 0.2, 0.5)):
        label = r"$1-\delta$" if index == 0 else ""
        lines.append(
            rf"{label} & {1.0-delta:.2f} & "
            rf"{cells(lookup.get(('gate_delta', delta)))} \\"
        )
    return "\n".join(lines)


def replace_table(text: str, rows: str) -> str:
    label_index = text.index(TABLE_LABEL)
    start = text.index(r"\midrule", label_index) + len(r"\midrule")
    end = text.index(r"\bottomrule", start)
    return text[:start] + "\n" + rows + "\n" + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper" / "main.tex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = completed_seed_logs(args.logs_dir)
    rows = make_rows(aggregate(paths))
    if args.dry_run:
        print(rows)
        return 0
    text = args.paper.read_text(encoding="utf-8")
    args.paper.write_text(replace_table(text, rows), encoding="utf-8")
    print(f"Updated {args.paper} from seeds {sorted(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
