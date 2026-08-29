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
      均值/UCB 冲突率    —— "TIP subspace" 行按全部 epoch 累计（rate×steps 反推计数求和）
      介入相对原更新改善率、相对当前预测损失改善率 —— 从 step 级数据识别投影步：
      优先 projection_applied 字段，回退 tip_projection_gain != 0（投影步
      的 gain 必非零）。改善率 = 投影步中 gain > 0 的比例；相对改善率 =
      100% × tip_projection_gain / pred_loss_before 的均值 ± 标准差，其中
      tip_projection_gain = raw_adamw_pred_loss_delta - pred_loss_delta。
  * 无日志的组合保留 \\ldots 占位。B1 行 TIP 相关列 (7-10) 恒为 --。
  * 输出直接改写 paper/main.tex 中带有 tab:key_metrics_template 标签的表格，
    采用“预测模型”和“方法”分列的 11 列 longtable 格式。
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
NATURAL_MISSING_DATASETS = {"Smart Manufacturing", "MetroPT-3"}
DATASET_DISPLAY = {
    "EWELD Load": r"\makecell{EWELD\\Load}",
    "Smart Manufacturing": r"\makecell{Smart\\Manufacturing}",
}
IMPUTER_DISPLAY = {
    "ModernTCN": r"\makecell{Modern\\TCN}",
}

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
    r"pred_loss_before=(?P<before>[-+\d.eE]+) \| "
    r"raw_adamw_pred_loss_delta=(?P<raw_delta>[-+\d.eE]+) \| "
    r"pred_loss_delta=(?P<delta>[-+\d.eE]+) \| "
    r"pred_loss_relative_delta=[-+\d.eE]+ \| "
    r"tip_projection_gain=(?P<gain>[-+\d.eE]+)"
    r"(?: \| projection_applied=(?P<projected>[01]))?"
)
EVAL_MSE_RE = re.compile(
    r"Evaluation Results Summary: split=test\b.*?"
    r"\bMSE\s+(?P<imp>[-+\d.eE]+|N/A)\s+(?P<pred>[-+\d.eE]+|N/A)", re.S
)
TIP_SUBSPACE_RE = re.compile(
    r"TIP subspace: raw_conflict_rate=(?P<raw>[-\d.eE]+) \| "
    r"(?:ucb_conflict_rate=(?P<ucb>[-\d.eE]+) \| )?"
    r"projection_rate=(?P<proj>[-\d.eE]+) \| steps=(?P<steps>\d+)"
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
        # 自然缺失数据集的 test 无完整真值：imp 列可能为 N/A，pred 列独立提取
        result["imp_mse"] = (float(m.group("imp"))
                             if m.group("imp") != "N/A" else None)
        result["pred_mse"] = (float(m.group("pred"))
                              if m.group("pred") != "N/A" else None)

    # 预测误差非增率：全部 step 记录中 pred_loss_delta <= 0 的比例
    steps = []
    for sm in STEP_RE.finditer(text):
        steps.append({
            "before": float(sm.group("before")),
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
            relative_gains = [
                100.0 * s["gain"] / s["before"]
                for s in proj_steps if s["before"] > 1e-12
            ]
            if relative_gains:
                mean = sum(relative_gains) / len(relative_gains)
                std = (
                    sum((gain - mean) ** 2 for gain in relative_gains)
                    / len(relative_gains)
                ) ** 0.5
                result["relative_current_loss_gain_mean"] = mean
                result["relative_current_loss_gain_std"] = std

    # TIP subspace：全部 epoch 累计（每 epoch 一条，含该 epoch 的 rate 与
    # steps；按 rate×steps 反推计数后累加，再除以总步数——反映整个训练过程，
    # 而非最后一个 epoch 的快照）
    subspaces = list(TIP_SUBSPACE_RE.finditer(text))
    if subspaces:
        raw_total = 0
        ucb_total = 0
        proj_total = 0
        steps_total = 0
        for match in subspaces:
            steps_i = int(match.group("steps"))
            raw_total += round(float(match.group("raw")) * steps_i)
            if match.group("ucb") is not None:
                ucb_total += round(float(match.group("ucb")) * steps_i)
            proj_total += round(float(match.group("proj")) * steps_i)
            steps_total += steps_i
        if steps_total > 0:
            result["raw_conflict_rate"] = raw_total / steps_total * 100.0
            if "ucb_total" in locals():
                result["ucb_conflict_rate"] = ucb_total / steps_total * 100.0
            result["projection_rate"] = proj_total / steps_total * 100.0

    if verbose:
        print(f"  {path.name}: imp={result['imp_mse']} pred={result['pred_mse']} "
              f"noninc={result['nonincrease_rate']} raw={result['raw_conflict_rate']} "
              f"ucb={result['ucb_conflict_rate']}")
    return result


def fmt_mse(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else r"\ldots"


def fmt_pct(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else r"\ldots"


def fmt_gain(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "--"
    return f"${mean:.4f}\\pm{std:.4f}$"


def emphasize(value: str, is_best: bool) -> str:
    """标粗标红同一 B1/B2 配对中的较优值。"""
    return rf"\textcolor{{red}}{{\textbf{{{value}}}}}" if is_best else value


def pairwise_best(
    b1: dict | None,
    b2: dict | None,
    key: str,
    *,
    higher_is_better: bool,
    digits: int,
) -> dict[str, bool]:
    """按表格显示精度判定 B1/B2 最优；缺值时不进行比较。"""
    flags = {"B1": False, "B2": False}
    if b1 is None or b2 is None:
        return flags
    value1 = b1.get(key)
    value2 = b2.get(key)
    if value1 is None or value2 is None:
        return flags
    shown1 = round(value1, digits)
    shown2 = round(value2, digits)
    best = max(shown1, shown2) if higher_is_better else min(shown1, shown2)
    flags["B1"] = shown1 == best
    flags["B2"] = shown2 == best
    return flags


def build_cells(
    exp_type: str,
    dataset: str,
    data: dict | None,
    best: dict[str, bool],
) -> str:
    """返回第 5--11 列的 LaTeX 单元格（B1/B2 共用）。"""
    imp_missing = "--" if dataset in NATURAL_MISSING_DATASETS else r"\ldots"
    if data is None:
        # 无日志：B1 有 MSE/非增率占位 + --；B2 全占位
        if exp_type == "B1":
            return (f"{imp_missing} & \\ldots & \\ldots & "
                    "-- & -- & -- & --")
        return (f"{imp_missing} & \\ldots & \\ldots & \\ldots & "
                "\\ldots & \\ldots & \\ldots")
    imp = ("--" if dataset in NATURAL_MISSING_DATASETS
           else fmt_mse(data["imp_mse"]))
    pred = fmt_mse(data["pred_mse"])
    noninc = fmt_pct(data["nonincrease_rate"])
    imp = emphasize(imp, best["imp_mse"])
    pred = emphasize(pred, best["pred_mse"])
    noninc = emphasize(noninc, best["nonincrease_rate"])
    if exp_type == "B1":
        return f"{imp} & {pred} & {noninc} & -- & -- & -- & --"
    raw = fmt_pct(data["raw_conflict_rate"])
    ucb = fmt_pct(data.get("ucb_conflict_rate"))
    inter = (fmt_pct(data.get("intervention_improve_rate"))
             if data.get("intervention_improve_rate") is not None else "--")
    gain = fmt_gain(data.get("relative_current_loss_gain_mean"),
                    data.get("relative_current_loss_gain_std"))
    return f"{imp} & {pred} & {noninc} & {raw} & {ucb} & {inter} & {gain}"


def generate_table(results: dict, verbose: bool = False) -> str:
    rows = []
    for di, ds in enumerate(DATASETS):
        for ii, imp in enumerate(IMPUTERS):
            for pi, pred in enumerate(PREDICTORS):
                b1 = results.get(("B1", ds, imp, pred))
                b2 = results.get(("B2", ds, imp, pred))
                best_by_metric = {
                    "imp_mse": pairwise_best(
                        b1, b2, "imp_mse", higher_is_better=False, digits=4,
                    ),
                    "pred_mse": pairwise_best(
                        b1, b2, "pred_mse", higher_is_better=False, digits=4,
                    ),
                    "nonincrease_rate": pairwise_best(
                        b1, b2, "nonincrease_rate",
                        higher_is_better=True, digits=2,
                    ),
                }
                if b1 is None and b2 is None and verbose:
                    print(f"  缺日志: {ds} / {imp} / {pred}（B1 与 B2）")
                for gi, (exp_type, data) in enumerate(
                    (("B1", b1), ("B2", b2))
                ):
                    if ii == 0 and pi == 0 and exp_type == "B1":
                        ds_display = DATASET_DISPLAY.get(ds, ds)
                        ds_cell = f"\\multirow{{36}}{{*}}{{{ds_display}}}"
                        imp_display = IMPUTER_DISPLAY.get(imp, imp)
                        imp_cell = f"\\multirow{{6}}{{*}}{{{imp_display}}}"
                    else:
                        ds_cell = ""
                        imp_cell = (
                            f"\\multirow{{6}}{{*}}{{{IMPUTER_DISPLAY.get(imp, imp)}}}"
                            if pi == 0 and exp_type == "B1"
                            else ""
                        )
                    best = {
                        metric: flags[exp_type]
                        for metric, flags in best_by_metric.items()
                    }
                    cells = build_cells(exp_type, ds, data, best)
                    is_block_end = (
                        ii == len(IMPUTERS) - 1
                        and pi == len(PREDICTORS) - 1
                        and exp_type == "B2"
                    )
                    rows.append(
                        f"{ds_cell} & {imp_cell} & {pred} & {exp_type} & "
                        f"{cells} \\\\{'*' if not is_block_end else ''}"
                    )
                if pi < len(PREDICTORS) - 1:
                    rows.append("\\cmidrule(l){3-11}")
            if ii < len(IMPUTERS) - 1:
                rows.append("\\cmidrule(l){2-11}")
        if di < len(DATASETS) - 1:
            rows.append("\\midrule")
    return "\n".join(rows)


HEADER = (
    "数据集 & 插补模型 & 预测模型 & 方法 &\n"
    "\\makecell{终点插补\\\\MSE} &\n"
    "\\makecell{终点预测\\\\MSE} &\n"
    "\\makecell{预测误差\\\\非增率（\\%）} &\n"
    "\\makecell{均值冲突\\\\判断率（\\%）} &\n"
    "\\makecell{UCB 冲突\\\\判断率（\\%）} &\n"
    "\\makecell{介入相对原更新\\\\改善率（\\%）} &\n"
    "\\makecell{相对当前预测损失改善率\\\\均值 $\\pm$ 标准差（\\%）} \\\\"
)


def build_table_block(body: str) -> str:
    return (
        "\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\renewcommand{\\arraystretch}{1.0}\n"
        f"\\begin{{longtable}}{{llllccccccc}}\n"
        f"\\caption{{真实场景验证：关键汇报量代填表}}"
        f"\\label{{tab:key_metrics_template}}\\\\\n"
        "\\toprule\n" + HEADER + "\n\\midrule\n\\endfirsthead\n"
        "\\multicolumn{11}{l}{\\textit{（续表）}}\\\\\n"
        "\\toprule\n" + HEADER + "\n\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endlastfoot\n" + body + "\n\\end{longtable}\n"
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
    label = "\\label{tab:key_metrics_template}"
    label_pos = tex.index(label)
    start = tex.rfind("\\begingroup", 0, label_pos)
    if start < 0:
        raise ValueError(f"未找到 {label} 前的 \\begingroup")
    table_end = tex.index("\\end{longtable}", label_pos) + len("\\end{longtable}")
    endgroup_pos = tex.find("\\endgroup", table_end)
    if endgroup_pos < 0:
        raise ValueError(f"未找到 {label} 表格后的 \\endgroup")
    end = endgroup_pos + len("\\endgroup")
    args.paper.write_text(tex[:start] + block + tex[end:], encoding="utf-8")
    print(f"已写入 {args.paper}（{sum(1 for _ in results if _[0]=='B2')} 个 B2 配置）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
