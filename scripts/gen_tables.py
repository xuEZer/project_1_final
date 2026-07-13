#!/usr/bin/env python3
"""Generate LaTeX tables for the paper by parsing experiment logs.

Usage:
  python scripts/gen_tables.py              # print all tables to stdout
  python scripts/gen_tables.py --table main  # only main results table
  python scripts/gen_tables.py --table grid  # only grid search table
  python scripts/gen_tables.py --table ablation  # only ablation table
  python scripts/gen_tables.py --table robustness  # only robustness table
  python scripts/gen_tables.py --table cost  # only compute cost table
  python scripts/gen_tables.py --table all --output tables.tex  # write to file
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import resolve_experiment_name
LOG_DIR = ROOT / "logs"
BEST_CFG_DIR = ROOT / "configs" / "best_cfg"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    imp_mae: float = 0.0
    imp_mse: float = 0.0
    pred_mae: float = 0.0
    pred_mse: float = 0.0
    best_val_loss: float | None = None
    completed: bool = False


@dataclass
class BestConfig:
    lr: str = ""
    bs: str = ""
    ckpt: str = ""


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_log(path: Path) -> EvalResult:
    text = path.read_text()
    r = EvalResult()

    val_losses = []
    for m in re.finditer(r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+)", text):
        val_losses.append(float(m.group(3)))
    if val_losses:
        r.best_val_loss = min(val_losses)

    for line in text.splitlines():
        m = re.search(r"\b(MAE|MSE)\b\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            metric, imp, pred = m.group(1), float(m.group(2)), float(m.group(3))
            if metric == "MAE":
                r.imp_mae, r.pred_mae = imp, pred
            elif metric == "MSE":
                r.imp_mse, r.pred_mse = imp, pred

    r.completed = "Experiment completed successfully!" in text
    return r


def load_all_results() -> dict[str, EvalResult]:
    results: dict[str, EvalResult] = {}
    for log_path in sorted(LOG_DIR.glob("*.log")):
        name = log_path.stem
        r = parse_log(log_path)
        if r.completed:
            results[name] = r
    return results


def load_best_configs() -> dict[str, BestConfig]:
    cfgs: dict[str, BestConfig] = {}
    for cfg_path in sorted(BEST_CFG_DIR.glob("best_*.ini")):
        ds_imp = cfg_path.stem[5:]  # remove "best_"
        bc = BestConfig()
        section = ""
        for line in cfg_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            elif "=" in line and section:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if section == "training" and k == "lr":
                    bc.lr = v
                elif section == "loader" and k == "batch_size":
                    bc.bs = v
                elif section == "paths" and k == "ckpt":
                    bc.ckpt = v
        if bc.lr and bc.bs:
            cfgs[ds_imp] = bc
    return cfgs


# ---------------------------------------------------------------------------
# Dynamic discovery of datasets / imputers / predictors
# ---------------------------------------------------------------------------

MAIN_EXP_JSON = ROOT / "configs" / "main_exp.json"

# Short display names for datasets in LaTeX tables
_DATASET_LABELS: dict[str, str] = {
    "Supply_Chain_Customer_D": "Supply Chain",
    "current_velocity_H": "Current Vel.",
    "azure2019_D_5T": "Azure2019-D",
    "Coastal_T_S_15T": "Coastal T-S",
    "CPHL_30T": "CPHL",
    "MetroPT-3_5T": "MetroPT-3",
    "Australia_Solar_H": "Aus. Solar",
}

LRS = ["0.001", "0.0005", "0.0001"]
BSS = ["16", "32", "64"]


def _load_main_exp() -> dict:
    """Load main_exp.json or return empty dict."""
    if MAIN_EXP_JSON.exists():
        with open(MAIN_EXP_JSON) as f:
            return json.load(f)
    return {}


def discover() -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Return (datasets, imputers, predictors) strictly from main_exp.json."""
    cfg = _load_main_exp()
    cfg_datasets = cfg.get("datasets", [])
    cfg_imputers = cfg.get("imputers", [])
    cfg_predictors = cfg.get("predictors", [])

    datasets = [(d, _DATASET_LABELS.get(d, d)) for d in cfg_datasets]
    return datasets, list(cfg_imputers), list(cfg_predictors)


# Module-level singletons (lazily resolved on first use)
_DATASETS: list[tuple[str, str]] | None = None
_IMPUTERS: list[str] | None = None
_PREDICTORS: list[str] | None = None


def _init() -> None:
    global _DATASETS, _IMPUTERS, _PREDICTORS
    if _DATASETS is None:
        _DATASETS, _IMPUTERS, _PREDICTORS = discover()


def DATASETS() -> list[tuple[str, str]]:
    _init()
    return _DATASETS  # type: ignore[return-value]


def IMPUTERS() -> list[str]:
    _init()
    return _IMPUTERS  # type: ignore[return-value]


def PREDICTORS() -> list[str]:
    _init()
    return _PREDICTORS  # type: ignore[return-value]


def _match_log(
    results: dict[str, EvalResult],
    exp_type: str,
    ds_key: str,
    imp: str,
    pred: str,
    lr: str,
    bs: str,
    extra: str = "",
) -> EvalResult | None:
    """Find the exact run defined by the current global and template configs."""
    overrides = [
        "--experiment.type", exp_type,
        "--data.dataset", ds_key,
        "--imputer.model", imp,
        "--predictor.model", pred,
        "--training.lr", lr,
        "--loader.batch_size", bs,
    ]

    k_match = re.search(r"(?:^|_)k([\d.]+)(?:_|$)", extra)
    if k_match:
        overrides.extend(["--tip.k", k_match.group(1)])

    p_match = re.search(r"(?:^|_)p([\d.]+)(?:_|$)", extra)
    if p_match:
        missing_rate = p_match.group(1)
        overrides.extend([
            "--data.missing_rate", missing_rate,
            "--data.artificial_missing_rate", missing_rate,
        ])

    return results.get(_expected_experiment_name(tuple(overrides)))


@lru_cache(maxsize=None)
def _expected_experiment_name(overrides: tuple[str, ...]) -> str:
    """Generate a canonical name using the same config path as main.py."""
    return resolve_experiment_name(list(overrides))


def fmt_val(v: float | None, best: bool = False) -> str:
    if v is None:
        return r"\ldots"
    s = f"{v:.4f}"
    return rf"\best{{{s}}}" if best else s


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def gen_grid_search_table(results: dict[str, EvalResult],
                          best_cfgs: dict[str, BestConfig]) -> str:
    """Generate the hyperparameter grid search table (tab:hparam_grid)."""

    def _row(ds_key: str, ds_label: str, imp: str) -> str:
        ds_imp = f"{ds_key}_{imp}"
        bc = best_cfgs.get(ds_imp)
        best_lr = bc.lr if bc else ""
        best_bs = bc.bs if bc else ""

        cells = []
        for lr in LRS:
            for bs in BSS:
                r = _match_log(results, "B1", ds_key, imp, "Chronos", lr, bs)
                if r is None:
                    # Try other predictors (val_loss should be same)
                    for pred in PREDICTORS():
                        r = _match_log(results, "B1", ds_key, imp, pred, lr, bs)
                        if r:
                            break
                is_best = (lr == best_lr and bs == best_bs)
                cells.append(fmt_val(r.best_val_loss if r else None, is_best))
        return f"  {ds_label:<22s} & {' & '.join(cells)} \\\\"

    tables = []
    for imp in IMPUTERS():
        rows = []
        for ds_key, ds_label in DATASETS():
            rows.append(_row(ds_key, ds_label, imp))

        table = rf"""% ====== ({chr(97 + IMPUTERS().index(imp))}) {imp} ======
\textbf{{({chr(97 + IMPUTERS().index(imp))}) {imp}}}

\vspace{{4pt}}
\begin{{tabular}}{{lccc|ccc|ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{$\text{{LR}}=10^{{-3}}$}} & \multicolumn{{3}}{{c}}{{$5\!\times\!10^{{-4}}$}} & \multicolumn{{3}}{{c}}{{$10^{{-4}}$}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}
数据集 & BS=16 & 32 & 64 & 16 & 32 & 64 & 16 & 32 & 64 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""
        tables.append(table)

    return "\n\n\\vspace{10pt}\n\n".join(tables)


def gen_main_results_table(results: dict[str, EvalResult],
                           best_cfgs: dict[str, BestConfig]) -> str:
    """Generate the main prediction results table (tab:main_results)."""

    # Build data matrix: [ds_idx][imp_idx][pred_idx] -> (b1_mse, b1_mae, b2_mse, b2_mae)
    data: dict = {}
    for _ds_idx, (ds_key, _) in enumerate(DATASETS()):
        for imp in IMPUTERS():
            ds_imp = f"{ds_key}_{imp}"
            bc = best_cfgs.get(ds_imp)
            if not bc:
                continue
            for pred in PREDICTORS():
                b1 = _match_log(results, "B1", ds_key, imp, pred, bc.lr, bc.bs)
                b2 = _match_log(results, "B2", ds_key, imp, pred, bc.lr, bc.bs)
                data[(ds_key, imp, pred)] = (b1, b2)

    def _b1_b2_cells(b1: EvalResult | None, b2: EvalResult | None) -> tuple[str, str, str, str]:
        b1_mse = fmt_val(b1.pred_mse if b1 else None)
        b1_mae = fmt_val(b1.pred_mae if b1 else None)
        if b2 and b1 and b2.pred_mse < b1.pred_mse:
            b2_mse = fmt_val(b2.pred_mse, True)
            b2_mae = fmt_val(b2.pred_mae, True)
        else:
            b2_mse = fmt_val(b2.pred_mse if b2 else None)
            b2_mae = fmt_val(b2.pred_mae if b2 else None)
        return b1_mse, b1_mae, b2_mse, b2_mae

    # --- (a) Sundial: full table ---
    sundial_rows = []
    for ds_key, ds_label in DATASETS():
        ds_imp_rows = []
        for imp in IMPUTERS():
            b1, b2 = data.get((ds_key, imp, "Sundial"), (None, None))
            mse1, mae1, mse2, mae2 = _b1_b2_cells(b1, b2)
            ds_imp_rows.append(f"                              & {imp:<8s} & {mse1} & {mae1} & {mse2} & {mae2} \\\\")
        if ds_imp_rows:
            n_imps = len(ds_imp_rows)
            first = ds_imp_rows[0].replace("                              &", rf"\multirow{{{n_imps}}}{{*}}{{{ds_label}}} &", 1)
            ds_imp_rows[0] = first
            sundial_rows.extend(ds_imp_rows)
            sundial_rows.append(r"\midrule")

    # Remove trailing \midrule
    if sundial_rows and sundial_rows[-1] == r"\midrule":
        sundial_rows.pop()

    sundial_table = rf"""% ====== (a) Sundial ======
\begin{{minipage}}[t]{{0.48\textwidth}}
\centering
\textbf{{(a) Sundial}}
\vspace{{4pt}}
\begin{{tabular}}{{llcccc}}
\toprule
\multirow{{2}}{{*}}{{数据集}} & \multirow{{2}}{{*}}{{插补模型}} & \multicolumn{{2}}{{c}}{{B1}} & \multicolumn{{2}}{{c}}{{B2}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
& & MSE & MAE & MSE & MAE \\
\midrule
{chr(10).join(sundial_rows)}
\bottomrule
\end{{tabular}}
\end{{minipage}}"""

    # --- (b) Timer, (c) Chronos: compact tables ---
    compact_tables = []
    for pred_idx, pred in enumerate(["Timer", "Chronos"]):
        label = ["(b) Timer", "(c) Chronos"][pred_idx]
        rows = []
        for ds_key, _ in DATASETS():
            for imp in IMPUTERS():
                b1, b2 = data.get((ds_key, imp, pred), (None, None))
                mse1, mae1, mse2, mae2 = _b1_b2_cells(b1, b2)
                rows.append(f"{mse1} & {mae1} & {mse2} & {mae2} \\\\")
            if rows:
                rows.append(r"\midrule")
        if rows and rows[-1] == r"\midrule":
            rows.pop()

        compact_tables.append(rf"""% ====== {label} ======
\begin{{minipage}}[t]{{0.24\textwidth}}
\centering
\textbf{{{label}}}
\vspace{{4pt}}
\begin{{tabular}}{{cccc}}
\toprule
\multicolumn{{2}}{{c}}{{B1}} & \multicolumn{{2}}{{c}}{{B2}} \\
\cmidrule(lr){{1-2}}\cmidrule(lr){{3-4}}
MSE & MAE & MSE & MAE \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{minipage}}""")

    return f"""{sundial_table}
\hfill
{compact_tables[0]}
\hfill
{compact_tables[1]}"""


def gen_ablation_table(results: dict[str, EvalResult],
                       best_cfgs: dict[str, BestConfig]) -> str:
    """Generate the projection ablation table (tab:projection_ablation)."""

    # Only Supply Chain and Current Velocity, SAITS + DeepMVI, Timer + Chronos
    ds_list = [("Supply_Chain_Customer_D", "Supply Chain"),
               ("current_velocity_H", "Current Vel.")]
    imp_list = ["SAITS", "DeepMVI"]
    pred_list = ["Timer", "Chronos"]
    strategies = ["B1", "B2", "B4", "B5", "B6"]

    rows = []
    for ds_key, ds_label in ds_list:
        first_row_of_ds = True
        for imp in imp_list:
            first_row_of_imp = True
            for pred in pred_list:
                ds_imp = f"{ds_key}_{imp}"
                bc = best_cfgs.get(ds_imp)
                lr = bc.lr if bc else "?"
                bs = bc.bs if bc else "?"

                cells = []
                for st in strategies:
                    r = _match_log(results, st, ds_key, imp, pred, lr, bs)
                    cells.append(fmt_val(r.pred_mse if r else None))

                ds_cell = rf"\multirow{{2}}{{*}}{{{ds_label}}}" if first_row_of_ds else ""
                imp_cell = rf"\multirow{{2}}{{*}}{{{imp}}}" if first_row_of_imp else ""
                first_row_of_ds = False
                first_row_of_imp = not first_row_of_imp

                rows.append(f"{ds_cell} & {imp_cell} & {pred:<7s} & {' & '.join(cells)} \\\\")
            rows.append(r"\midrule")

    if rows and rows[-1] == r"\midrule":
        rows.pop()

    return rf"""\begin{{tabular}}{{lllccccc}}
\toprule
数据集 & 插补模型 & 预测模型 & B1 & B2 & B4 & B5 & B6 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""


def gen_robustness_table(results: dict[str, EvalResult],
                         best_cfgs: dict[str, BestConfig]) -> str:
    """Generate the missing rate robustness table (tab:missing_robustness)."""

    # Fixed: Supply Chain, SAITS, Chronos
    ds_key = "Supply_Chain_Customer_D"
    imp = "SAITS"
    pred = "Chronos"
    ds_imp = f"{ds_key}_{imp}"
    bc = best_cfgs.get(ds_imp)
    lr = bc.lr if bc else "?"
    bs = bc.bs if bc else "?"

    ps = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    rows = []
    for p in ps:
        cells = []
        for st in ["B1", "B2", "B3"]:
            r = _match_log(results, st, ds_key, imp, pred, lr, bs, extra=f"p{p}")
            cells.append(fmt_val(r.pred_mse if r else None))
        rows.append(f"${p}$ & {' & '.join(cells)} \\\\")

    return rf"""\begin{{tabular}}{{cccc}}
\toprule
$p$ & B1 & B2 & B3 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""


def gen_cost_table(results: dict[str, EvalResult],
                   best_cfgs: dict[str, BestConfig]) -> str:
    """Generate the compute cost table (tab:compute_cost)."""

    # Fixed: Supply Chain, SAITS, Chronos
    ds_key = "Supply_Chain_Customer_D"
    imp = "SAITS"
    pred = "Chronos"
    ds_imp = f"{ds_key}_{imp}"
    bc = best_cfgs.get(ds_imp)
    lr = bc.lr if bc else "?"
    bs = bc.bs if bc else "?"

    def _epoch_time(exp_type: str, k: str = "") -> float | None:
        extra = f"k{k}_" if k else ""
        r = _match_log(results, exp_type, ds_key, imp, pred, lr, bs, extra=extra)
        if r is None:
            return None
        # Read log to get timing
        prefix = f"{exp_type}_{ds_key}_{imp}_{pred}_lr{lr}_bs{bs}"
        if extra:
            prefix += f"_{extra}"
        for log_path in sorted(LOG_DIR.glob(f"{prefix}*.log")):
            if log_path.is_file():
                text = log_path.read_text()
                lines = text.splitlines()
                first_ts, last_ts = None, None
                for line in lines:
                    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    if m:
                        if first_ts is None:
                            first_ts = m.group(1)
                        last_ts = m.group(1)
                epoch_count = len([l for l in lines if "Epoch " in l and "/" in l])
                if first_ts and last_ts and epoch_count:
                    fmt = "%Y-%m-%d %H:%M:%S"
                    t1 = datetime.datetime.strptime(first_ts, fmt)
                    t2 = datetime.datetime.strptime(last_ts, fmt)
                    return (t2 - t1).total_seconds() / epoch_count
        return None

    b1_time = _epoch_time("B1")

    rows = []
    # B1 row
    rows.append(r"B1            & 1          & 1           & 0            & $1\times$               \\")
    # B2 with different k
    for k in ["5", "10", "20"]:
        t = _epoch_time("B2", k)
        rel = f"${t/b1_time:.1f}\\times$" if t and b1_time else r"\ldots"
        fwd = 1 + 2 * int(k)
        rows.append(rf"B2 ($k={k}$)    & {fwd:<10} & 1           & {fwd:<13} & {rel} \\")
    # B6
    t = _epoch_time("B6", "20")
    rel = f"${t/b1_time:.1f}\\times$" if t and b1_time else r"\ldots"
    rows.append(rf"B6 ($k=20$)   & 41         & 0           & 41           & {rel} \\")

    return rf"""\begin{{tabular}}{{lcccc}}
\toprule
策略          & 插补前向次数 & 插补反向次数 & TSFM前向次数 & 相对耗时（B1=$1\times$） \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


TABLE_GENERATORS = {
    "grid": ("Hyperparameter Grid Search", gen_grid_search_table),
    "main": ("Main Prediction Results", gen_main_results_table),
    "ablation": ("Projection Ablation", gen_ablation_table),
    "robustness": ("Missing Rate Robustness", gen_robustness_table),
    "cost": ("Compute Cost", gen_cost_table),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment logs")
    parser.add_argument("--table", choices=["all", "grid", "main", "ablation", "robustness", "cost"],
                        default="all", help="Which table to generate (default: all)")
    parser.add_argument("--output", "-o", type=Path,
                        help="Write output to file instead of stdout")
    parser.add_argument("--env", action="store_true",
                        help="Wrap output in a table environment (for standalone use)")
    args = parser.parse_args()

    results = load_all_results()
    best_cfgs = load_best_configs()

    if args.table == "all":
        tables_to_gen = list(TABLE_GENERATORS.items())
    else:
        tables_to_gen = [(args.table, TABLE_GENERATORS[args.table])]

    output_parts = []
    for key, (title, gen_fn) in tables_to_gen:
        header = f"% {'='*70}\n% {title}\n% {'='*70}"
        table_body = gen_fn(results, best_cfgs)

        if args.env:
            table_body = rf"""\begin{{table}}[H]
\centering
\caption{{{title}}}
\label{{tab:{key}}}
\footnotesize
{table_body}
\end{{table}}"""

        output_parts.append(f"{header}\n{table_body}")

    output = "\n\n".join(output_parts)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
