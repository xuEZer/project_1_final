#!/usr/bin/env python3
"""Fill the RQ3 mechanism table from shared-trajectory analysis logs."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

try:
    from fill_table6 import fmt_gain, fmt_pct
except ModuleNotFoundError:  # Imported as ``scripts.fill_table7`` in tests/tools.
    from scripts.fill_table6 import fmt_gain, fmt_pct


ROOT = Path(__file__).resolve().parent.parent
TABLE_LABEL = r"\label{tab:projection_ablation}"
STEP_RE = re.compile(
    r"RQ3 counterfactual step: step=(?P<step>\d+) \| variant=(?P<variant>B[1-5]) \| "
    r"pred_loss_before=(?P<before>[-+\d.eE]+) \| "
    r"candidate_pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"conflict=(?P<conflict>[01]) \| projected=(?P<projected>[01]) \| "
    r"gain_vs_raw=(?P<gain>[-+\d.eE]+) \| "
    r"relative_gain=(?P<relative>[-+\d.eE]+)"
)
SEED_RE = re.compile(
    r"Seed routing: experiment_seed=(?P<experiment>\d+) \| "
    r"missing_seed=(?P<missing>\d+)"
)
UID_RE = re.compile(r"Generated preprocess UID: (?P<uid>[0-9a-f]{16})")


def completed(path: Path) -> bool:
    return "Experiment completed successfully!" in path.read_text(
        encoding="utf-8", errors="replace"
    )


def parse_rq3(path: Path) -> dict[str, dict]:
    grouped = {stage: [] for stage in ("B1", "B2", "B3", "B4", "B5")}
    by_step: dict[int, dict[str, dict]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in STEP_RE.finditer(text):
        item = {
            "delta": float(match.group("delta")),
            "conflict": int(match.group("conflict")),
            "projected": int(match.group("projected")),
            "gain": float(match.group("gain")),
            "relative": float(match.group("relative")),
        }
        stage = match.group("variant")
        grouped[stage].append(item)
        by_step.setdefault(int(match.group("step")), {})[stage] = item
    results = {}
    for stage, steps in grouped.items():
        if not steps:
            continue
        result = {
            "nonincrease_rate": 100.0 * sum(
                step["delta"] <= 0.0 for step in steps
            ) / len(steps),
        }
        if stage != "B1":
            result["intervention_rate"] = 100.0 * sum(
                step["projected"] for step in steps
            ) / len(steps)
            safe_steps = [
                variants
                for variants in by_step.values()
                if "B1" in variants and variants["B1"]["delta"] <= 0.0
            ]
            if safe_steps:
                result["false_positive_rate"] = 100.0 * sum(
                    variants.get(stage, {}).get("projected", 0)
                    for variants in safe_steps
                ) / len(safe_steps)
            projected = [step for step in steps if step["projected"]]
            if projected:
                result["intervention_improve_rate"] = 100.0 * sum(
                    step["gain"] > 0.0 for step in projected
                ) / len(projected)
                gains = [100.0 * step["relative"] for step in projected]
                result["relative_current_loss_gain_mean"] = statistics.fmean(gains)
                result["relative_current_loss_gain_std"] = statistics.pstdev(gains)
        results[stage] = result
    return results


def collect(logs_dir: Path) -> dict[str, dict | None]:
    paths = list(logs_dir.glob("RQ3_azure2019_I_5T_SAITS_Toto2_*.log"))
    by_seed: dict[int, tuple[Path, str]] = {}
    for path in paths:
        if not completed(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        seed_match = SEED_RE.search(text)
        uid_match = UID_RE.search(text)
        if seed_match is None or uid_match is None:
            continue
        seed = int(seed_match.group("experiment"))
        if int(seed_match.group("missing")) != 42 or seed not in {42, 43, 44}:
            continue
        previous = by_seed.get(seed)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            by_seed[seed] = (path, uid_match.group("uid"))
    if set(by_seed) != {42, 43, 44}:
        raise FileNotFoundError(
            f"Expected completed RQ3 seeds 42, 43, 44; found {sorted(by_seed)}"
        )
    if len({uid for _, uid in by_seed.values()}) != 1:
        raise ValueError("RQ3 seed logs do not share one preprocessing UID")

    parsed = {seed: parse_rq3(path) for seed, (path, _) in by_seed.items()}
    output: dict[str, dict | None] = {}
    for stage in ("B1", "B2", "B3", "B4", "B5"):
        seed_results = [parsed[seed].get(stage) for seed in sorted(parsed)]
        if any(result is None for result in seed_results):
            output[stage] = None
            continue
        result = {}
        fields = (
            "nonincrease_rate",
            "intervention_rate",
            "false_positive_rate",
            "intervention_improve_rate",
        )
        for field in fields:
            values = [item[field] for item in seed_results if field in item]
            if values:
                result[field] = statistics.fmean(values)
        gains = [
            item["relative_current_loss_gain_mean"]
            for item in seed_results
            if "relative_current_loss_gain_mean" in item
        ]
        if gains:
            result["relative_current_loss_gain_mean"] = statistics.fmean(gains)
            result["relative_current_loss_gain_std"] = (
                statistics.stdev(gains) if len(gains) > 1 else 0.0
            )
        output[stage] = result
    return output


def row(stage: str, data: dict | None) -> str:
    if data is None:
        if stage == "B1":
            return rf"{stage} & \ldots & -- & -- & -- & -- \\*"
        return rf"{stage} & \ldots & \ldots & \ldots & \ldots & $\ldots\pm\ldots$ \\*"
    nonincrease = fmt_pct(data.get("nonincrease_rate"))
    if stage == "B1":
        return rf"{stage} & {nonincrease} & -- & -- & -- & -- \\*"
    intervention = fmt_pct(data.get("intervention_rate"))
    false_positive = fmt_pct(data.get("false_positive_rate"))
    improvement = fmt_pct(data.get("intervention_improve_rate"))
    gain = fmt_gain(
        data.get("relative_current_loss_gain_mean"),
        data.get("relative_current_loss_gain_std"),
    )
    return (
        rf"{stage} & {nonincrease} & {intervention} & {false_positive} & "
        rf"{improvement} & {gain} \\*"
    )


def replace_rows(text: str, results: dict[str, dict | None]) -> str:
    label_index = text.index(TABLE_LABEL)
    table_start = text.rfind(r"\begin{table}", 0, label_index)
    table_end = text.index(r"\end{table}", label_index)
    block = text[table_start:table_end]
    replacement = "\n".join(row(stage, results[stage]) for stage in results)
    pattern = re.compile(
        r"B1\s*&.*?\\\\\*\s*\n"
        r"B2\s*&.*?\\\\\*\s*\n"
        r"B3\s*&.*?\\\\\*\s*\n"
        r"B4\s*&.*?\\\\\*\s*\n"
        r"B5\s*&.*?\\\\\*",
        re.S,
    )
    updated, count = pattern.subn(lambda _: replacement, block, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the five RQ3 table rows")
    return text[:table_start] + updated + text[table_end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--paper", type=Path, default=ROOT / "paper" / "main.tex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = collect(args.logs_dir)
    output = replace_rows(args.paper.read_text(encoding="utf-8"), results)
    if args.dry_run:
        for stage, data in results.items():
            print(row(stage, data))
    else:
        args.paper.write_text(output, encoding="utf-8")
        print(f"Updated {args.paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
