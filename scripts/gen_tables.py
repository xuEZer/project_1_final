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


@dataclass
class EvalResult:
    imp_mae: float = 0.0
    imp_mse: float = 0.0
    pred_mae: float = 0.0
    pred_mse: float = 0.0
    best_val_loss: float | None = None
    val_imp_mse: float | None = None
    completed: bool = False


@dataclass
class BestConfig:
    lr: str = ""
    bs: str = ""
    ckpt: str = ""


def parse_log(path: Path) -> EvalResult:
    text = path.read_text()
    r = EvalResult()

    val_losses = []
    for m in re.finditer(r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+)", text):
        val_losses.append(float(m.group(3)))
    if val_losses:
        r.best_val_loss = min(val_losses)

    val_summary = re.search(
        r"Evaluation Results Summary: split=val(?P<body>.*?)"
        r"Evaluation completed on val split\.",
        text,
        flags=re.DOTALL,
    )
    if val_summary:
        val_mse = re.search(
            r"\bMSE\b\s+([-+\d.eE]+)\s+[-+\d.eE]+",
            val_summary.group("body"),
        )
        if val_mse:
            r.val_imp_mse = float(val_mse.group(1))

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
        ds_imp = cfg_path.stem[5:]
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


MAIN_EXP_JSON = ROOT / "configs" / "main_exp.json"

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
    if MAIN_EXP_JSON.exists():
        with open(MAIN_EXP_JSON) as f:
            return json.load(f)
    return {}


def discover() -> tuple[list[tuple[str, str]], list[str], list[str]]:
    cfg = _load_main_exp()
    cfg_datasets = cfg.get("datasets", [])
    cfg_imputers = cfg.get("imputers", [])
    cfg_predictors = cfg.get("predictors", [])

    datasets = [(d, _DATASET_LABELS.get(d, d)) for d in cfg_datasets]
    return datasets, list(cfg_imputers), list(cfg_predictors)


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
    overrides = [
        "--experiment.type", exp_type,
        "--data.dataset", ds_key,
        "--imputer.model", imp,
        "--predictor.model", pred,
        "--training.lr", lr,
        "--loader.batch_size", bs,
    ]

    rank_match = re.search(r"(?:^|_)rank([\d.]+)(?:_|$)", extra)
    if rank_match:
        overrides.extend(["--tip.subspace_rank", rank_match.group(1)])

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
    return resolve_experiment_name(list(overrides))


def fmt_val(v: float | None, best: bool = False) -> str:
    if v is None:
        return r"\ldots"
    s = f"{v:.4f}"
    return rf"\best{{{s}}}" if best else s


def gen_grid_search_table(results: dict[str, EvalResult],
                          best_cfgs: dict[str, BestConfig]) -> str:
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
                cells.append(fmt_val(r.val_imp_mse if r else None, is_best))
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

    compact_tables = []
    for pred_idx, pred in enumerate(["Toto2", "Chronos"]):
        label = ["(b) Toto2", "(c) Chronos"][pred_idx]
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
\\hfill
{compact_tables[0]}
\\hfill
{compact_tables[1]}"""


def gen_ablation_table(results: dict[str, EvalResult],
                       best_cfgs: dict[str, BestConfig]) -> str:
    # Only Supply Chain and Current Velocity, SAITS + DeepMVI, Toto2 + Chronos
    ds_list = [("Supply_Chain_Customer_D", "Supply Chain"),
               ("current_velocity_H", "Current Vel.")]
    imp_list = ["SAITS", "DeepMVI"]
    pred_list = ["Toto2", "Chronos"]
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
        for st in ["B1", "B2"]:
            r = _match_log(results, st, ds_key, imp, pred, lr, bs, extra=f"p{p}")
            cells.append(fmt_val(r.pred_mse if r else None))
        rows.append(f"${p}$ & {' & '.join(cells)} \\\\")

    return rf"""\begin{{tabular}}{{ccc}}
\toprule
$p$ & B1 & B2 \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""


def gen_cost_table(results: dict[str, EvalResult],
                   best_cfgs: dict[str, BestConfig]) -> str:
    # Fixed: Supply Chain, SAITS, Chronos
    ds_key = "Supply_Chain_Customer_D"
    imp = "SAITS"
    pred = "Chronos"
    ds_imp = f"{ds_key}_{imp}"
    bc = best_cfgs.get(ds_imp)
    lr = bc.lr if bc else "?"
    bs = bc.bs if bc else "?"

    def _epoch_time(exp_type: str, rank: str = "") -> float | None:
        extra = f"rank{rank}_" if rank else ""
        r = _match_log(results, exp_type, ds_key, imp, pred, lr, bs, extra=extra)
        if r is None:
            return None
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
    rows.append(r"B1            & 1          & 1           & 0            & $1\times$               \\")
    for rank in ["2", "3", "4"]:
        t = _epoch_time("B2", rank)
        rel = f"${t/b1_time:.1f}\\times$" if t and b1_time else r"\ldots"
        fwd = 1 + 2 * int(rank)
        rows.append(rf"B2 ($r={rank}$)    & {fwd:<10} & 1           & 1            & {rel} \\")

    return rf"""\begin{{tabular}}{{lcccc}}
\toprule
策略          & 插补前向次数 & 插补反向次数 & TSFM前向次数 & 相对耗时（B1=$1\times$） \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}"""


TABLE_GENERATORS = {
    "grid": ("Hyperparameter Grid Search", gen_grid_search_table),
    "main": ("Main Prediction Results", gen_main_results_table),
    "ablation": ("Projection Ablation", gen_ablation_table),
    "robustness": ("Missing Rate Robustness", gen_robustness_table),
    "cost": ("Compute Cost", gen_cost_table),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment logs")
    parser.add_argument("--table", choices=["all", *TABLE_GENERATORS],
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
