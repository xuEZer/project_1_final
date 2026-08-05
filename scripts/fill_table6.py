#!/usr/bin/env python3
"""从实验日志自动填充论文表 6（真实训练轨迹验证：关键汇报量代填表）。

用法:
    python scripts/fill_table6.py [--logs-dir logs] [--paper paper/main.tex]
                                   [--dry-run] [--verbose]

说明:
  * 按 exp_name 解析 (B1/B2, 数据集, 插补模型, 预测模型)，配对同组合的 B1/B2 日志。
  * 填充的指标:
      终点插补/预测 MSE —— evaluate(split=test) 的 MSE 行
      预测误差非增率     —— "Prediction optimizer step" 行中 pred_loss_delta <= 0 的比例
      均值/UCB 冲突率    —— B2 最后一个 epoch 的 "TIP subspace" 行
      介入后改善率、相对原更新改善比例 —— 从 step 级数据识别投影步：
      优先 projection_applied 字段，回退 tip_projection_gain != 0（投影步
      的 gain 必非零）。改善率 = 投影步中 gain > 0 的比例；改善比例 =
      投影步中 gain / |raw_delta| 的均值 ± 标准差。
  * 无日志的组合保留 \\ldots 占位。B1 行 TIP 相关列 (7-10) 恒为 --。
  * 输出直接改写 paper/main.tex 的表 6（保留 multirow 合并与分页结构）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 表 6 网格（顺序与现有表一致）
DATASETS = ["Azure2019-I", "EWELD Load", "Coastal T-S",
            "Smart Manufacturing", "MetroPT-3"]
IMPUTERS = ["BRITS", "SAITS", "DeepMVI", "ModernTCN", "TimesNet", "TSLANet"]
PREDICTORS = ["Sundial", "Toto2", "Chronos-2"]

# exp_name 片段 → 表 6 显示名
DATASET_MAP = {
    "azure2019_I_5T": "Azure2019-I",
    "EWELD_Load_15T": "EWELD Load",
    "Coastal_T_S_15T": "Coastal T-S",
    "Smart_Manufacturing_H": "Smart Manufacturing",
    "MetroPT-3_5T": "MetroPT-3",
}
PREDICTOR_MAP = {"Chronos": "Chronos-2", "Sundial": "Sundial", "Toto2": "Toto2"}
IMPUTER_SET = set(IMPUTERS)

STEP_RE = re.compile(
    r"Prediction optimizer step: step=\d+ \| "
    r"pred_loss_before=[-+\d.eE]+ \| "
    r"raw_adamw_pred_loss_delta=(?P<raw_delta>[-+\d.eE]+) \| "
    r"pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"pred_loss_relative_delta=[-+\d.eE]+ \| "
    r"tip_projection_gain=(?P<gain>[-+\d.eE]+)"
    r"(?: \| projection_applied=(?P<projected>[01]))?"
)
EVAL_MSE_RE = re.compile(
    r"Evaluation Results Summary: split=test\b.*?"
    r"\bMSE\s+([-+\d.eE]+)\s+([-+\d.eE]+)", re.S
)
TIP_SUBSPACE_RE = re.compile(
    r"TIP subspace: raw_conflict_rate=(?P<raw>[-\d.eE]+) \| "
    r"(?:ucb_conflict_rate=(?P<ucb>[-\d.eE]+) \| )?"
    r"projection_rate=(?P<proj>[-\d.eE]+) \| steps=\d+"
)


def parse_exp_name(filename: str) -> tuple[str, str, str, str] | None:
    """解析 B{1-5}_{dataset}_{imputer}_{predictor}_lr.._bs..[_rank..]__{hash}."""
    stem = filename[:-4] if filename.endswith(".log") else filename
    tokens = stem.split("_")
    if not tokens or not re.fullmatch(r"B[1-5]", tokens[0]):
        return None
    exp_type = tokens[0]
    # 找到插补模型与预测模型的位置（从已知集合匹配）
    imp_idx = next((i for i, t in enumerate(tokens) if t in IMPUTER_SET), None)
    if imp_idx is None or imp_idx + 1 >= len(tokens):
        return None
    pred = tokens[imp_idx + 1]
    if pred not in PREDICTOR_MAP:
        return None
    dataset = "_".join(tokens[1:imp_idx])
    if dataset not in DATASET_MAP:
        return None
    return exp_type, DATASET_MAP[dataset], tokens[imp_idx], PREDICTOR_MAP[pred]


def parse_log(path: Path, verbose: bool = False) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict = {"imp_mse": None, "pred_mse": None,
                    "nonincrease_rate": None, "raw_conflict_rate": None,
                    "ucb_conflict_rate": None, "projection_rate": None}

    m = EVAL_MSE_RE.search(text)
    if m:
        result["imp_mse"] = float(m.group(1))
        result["pred_mse"] = float(m.group(2))

    # 预测误差非增率：全部 step 记录中 pred_loss_delta <= 0 的比例
    steps = []
    for sm in STEP_RE.finditer(text):
        steps.append({
            "delta": float(sm.group("delta")),
            "raw_delta": float(sm.group("raw_delta")),
            "gain": float(sm.group("gain")),
            "projected": (int(sm.group("projected"))
                          if sm.group("projected") is not None else None),
        })
    if steps:
        result["nonincrease_rate"] = (
            sum(1 for s in steps if s["delta"] <= 0.0) / len(steps) * 100.0
        )
        # 投影步识别：优先显式 projection_applied；回退 tip_projection_gain != 0
        # （project=False 时 final_update == g_i，gain 恒为 0；投影时 gain 几乎
        #  必非零，因 TIP 修正后的损失恰好等于 raw 损失的概率可忽略）
        if all(s["projected"] is not None for s in steps):
            proj_steps = [s for s in steps if s["projected"] == 1]
        else:
            proj_steps = [s for s in steps if abs(s["gain"]) > 1e-12]
        if proj_steps:
            improved = [s for s in proj_steps if s["gain"] > 0.0]
            result["intervention_improve_rate"] = (
                len(improved) / len(proj_steps) * 100.0
            )
            ratios = [
                s["gain"] / abs(s["raw_delta"])
                for s in proj_steps if abs(s["raw_delta"]) > 1e-12
            ]
            if ratios:
                mean = sum(ratios) / len(ratios)
                std = (sum((r - mean) ** 2 for r in ratios) / len(ratios)) ** 0.5
                result["improve_ratio_mean"] = mean
                result["improve_ratio_std"] = std

    # TIP subspace：取最后一个 epoch（最终统计）
    subspaces = list(TIP_SUBSPACE_RE.finditer(text))
    if subspaces:
        last = subspaces[-1]
        result["raw_conflict_rate"] = float(last.group("raw")) * 100.0
        if last.group("ucb") is not None:
            result["ucb_conflict_rate"] = float(last.group("ucb")) * 100.0
        result["projection_rate"] = float(last.group("proj")) * 100.0

    if verbose:
        print(f"  {path.name}: imp={result['imp_mse']} pred={result['pred_mse']} "
              f"noninc={result['nonincrease_rate']} raw={result['raw_conflict_rate']} "
              f"ucb={result['ucb_conflict_rate']}")
    return result


def fmt_mse(v: float | None) -> str:
    return f"{v:.4g}" if v is not None else r"\ldots"


def fmt_pct(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else r"\ldots"


def fmt_ratio(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return r"$\ldots\pm\ldots$"
    return f"${mean:.3f}\\pm{std:.3f}$"


def build_cells(exp_type: str, data: dict | None) -> str:
    """返回列 4-10 的 LaTeX 单元格（B1/B2 共用）。"""
    if data is None:
        # 无日志：B1 有 MSE/非增率占位 + --；B2 全占位
        if exp_type == "B1":
            return r"\ldots & \ldots & \ldots & -- & -- & -- & --"
        return (r"\ldots & \ldots & \ldots & \ldots & \ldots & \ldots"
                r" & $\ldots\pm\ldots$")
    imp = fmt_mse(data["imp_mse"])
    pred = fmt_mse(data["pred_mse"])
    noninc = fmt_pct(data["nonincrease_rate"])
    if exp_type == "B1":
        return f"{imp} & {pred} & {noninc} & -- & -- & -- & --"
    raw = fmt_pct(data["raw_conflict_rate"])
    ucb = fmt_pct(data.get("ucb_conflict_rate"))
    inter = fmt_pct(data.get("intervention_improve_rate"))
    ratio = fmt_ratio(data.get("improve_ratio_mean"),
                      data.get("improve_ratio_std"))
    return f"{imp} & {pred} & {noninc} & {raw} & {ucb} & {inter} & {ratio}"


def generate_table(results: dict, verbose: bool = False) -> str:
    rows = []
    for di, ds in enumerate(DATASETS):
        for ii, imp in enumerate(IMPUTERS):
            for pi, pred in enumerate(PREDICTORS):
                b1 = results.get(("B1", ds, imp, pred))
                b2 = results.get(("B2", ds, imp, pred))
                if b1 is None and b2 is None and verbose:
                    print(f"  缺日志: {ds} / {imp} / {pred}（B1 与 B2）")
                for gi, (exp_type, data) in enumerate(
                    (("B1", b1), ("B2", b2))
                ):
                    if ii == 0 and pi == 0 and exp_type == "B1":
                        ds_cell = f"\\multirow{{36}}{{*}}{{{ds}}}"
                    else:
                        ds_cell = ""
                    if pi == 0 and exp_type == "B1":
                        imp_cell = f"\\multirow{{6}}{{*}}{{{imp}}}"
                    else:
                        imp_cell = ""
                    cells = build_cells(exp_type, data)
                    is_block_end = (
                        ii == len(IMPUTERS) - 1 and gi == 1
                    )
                    rows.append(
                        f"{ds_cell} & {imp_cell} & {pred} / {exp_type} & "
                        f"{cells} \\\\{'*' if not is_block_end else ''}"
                    )
            if ii < len(IMPUTERS) - 1:
                rows.append("\\cmidrule(l){3-10}")
        if di < len(DATASETS) - 1:
            rows.append("\\midrule")
    return "\n".join(rows)


HEADER = (
    "数据集 & 插补模型 & 预测模型/方法 &\n"
    "\\makecell{终点插补\\\\MSE} &\n"
    "\\makecell{终点预测\\\\MSE} &\n"
    "\\makecell{预测误差\\\\非增率（\\%）} &\n"
    "\\makecell{均值冲突\\\\判断率（\\%）} &\n"
    "\\makecell{UCB 冲突\\\\判断率（\\%）} &\n"
    "\\makecell{介入后预测误差\\\\改善率（\\%）} &\n"
    "\\makecell{相对原更新改善比例\\\\均值 $\\pm$ 标准差} \\\\"
)


def build_table_block(body: str) -> str:
    return (
        "\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
        "\\renewcommand{\\arraystretch}{1.0}\n"
        f"\\begin{{tabularx}}{{\\textwidth}}{{llXccccccc}}\n"
        f"\\caption{{真实训练轨迹验证：关键汇报量代填表（训练预算为 5 个 epoch）}}"
        f"\\label{{tab:key_metrics_template}}\\\\\n"
        "\\toprule\n" + HEADER + "\n\\midrule\n\\endfirsthead\n"
        "\\multicolumn{10}{l}{\\textit{（续表）}}\\\\\n"
        "\\toprule\n" + HEADER + "\n\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endlastfoot\n" + body + "\n\\end{tabularx}\n"
        "\\endgroup"
    )


def collect_logs(logs_dir: Path, verbose: bool) -> dict:
    results = {}
    for path in sorted(logs_dir.glob("*.log")):
        parsed = parse_exp_name(path.name)
        if parsed is None:
            if verbose:
                print(f"  跳过无法解析的文件名: {path.name}")
            continue
        exp_type, ds, imp, pred = parsed
        if verbose:
            print(f"解析 {path.name} -> {exp_type} / {ds} / {imp} / {pred}")
        results[(exp_type, ds, imp, pred)] = parse_log(path, verbose)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--paper", type=Path, default=Path("paper/main.tex"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印生成的表 6，不写回论文")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"日志目录不存在: {args.logs_dir}", file=sys.stderr)
        return 2

    results = collect_logs(args.logs_dir, args.verbose)
    if not results:
        print("未找到可解析的实验日志", file=sys.stderr)
        return 2

    body = generate_table(results, args.verbose)
    block = build_table_block(body)

    if args.dry_run:
        print(block)
        return 0

    tex = args.paper.read_text(encoding="utf-8")
    start = tex.index("\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
                      "\\renewcommand{\\arraystretch}{1.0}\n"
                      "\\begin{tabularx}{\\textwidth}{llXccccccc}")
    end = tex.index("\\end{tabularx}\n\\endgroup") + len("\\end{tabularx}\n\\endgroup")
    args.paper.write_text(tex[:start] + block + "\n" + tex[end:], encoding="utf-8")
    print(f"已写入 {args.paper}（{sum(1 for _ in results if _[0]=='B2')} 个 B2 配置）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
