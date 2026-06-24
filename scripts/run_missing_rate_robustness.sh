#!/usr/bin/env bash
#
# 缺失率鲁棒性实验 — 论文 Section 5.5
#
# 用法:
#   ./scripts/run_missing_rate_robustness.sh                 # 运行完整实验矩阵
#   ./scripts/run_missing_rate_robustness.sh --dry-run       # 只打印命令
#   ./scripts/run_missing_rate_robustness.sh --failed-only   # 只重跑失败的实验
#
# 实验设计 (论文 Section 5.5):
#   - 数据集: Azure2019-D (高频工业), Supply Chain-Customer (低频商业)
#   - 插补模型: SAITS (注意力), DeepMVI (混合架构)
#   - 预测模型: Sundial, TimesFM25
#   - 缺失率: p ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
#   - 策略: B1 (基线), B2 (TIP), B3 (预训练+TIP)
#
# 关键假设:
#   - H1 (倒 U 形增益曲线): B2 改善率在 p=0.2~0.3 时最高
#   - H2 (B3 高缺失补救): p≥0.3 时 B3 逐渐优于 B2,差距随 p 增大

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------------------------------------------------------------------------
# 实验矩阵 — 论文 Section 5.5
# ---------------------------------------------------------------------------

# 两个领域特征差异最大的数据集
DATASETS=(
    "Supply_Chain_Customer_D"   # 低频商业供应链 (日粒度)
    "azure2019_D_5T"             # 高频工业系统监测 (5分钟粒度)
)

# 两种代表性插补架构
IMPUTATORS=(SAITS DeepMVI)

# 两种 TSFM
PREDICTORS=(Sundial TimesFM25)

# 缺失率五档
MISSING_RATES=(0.1 0.2 0.3 0.4 0.5)

# 策略: B1 基线, B2 TIP, B3 预训练+TIP
EXP_TYPES=(B1 B2 B3)

# 固定参数
LOOKBACK_LEN=128
SEEDS=(42)
TIP_K=20
TIP_EPSILON="1e-4"
TIP_PRETRAIN_EPOCHS=2

declare -A PYTHON_BIN=(
    ["Sundial"]="${PROJECT_DIR}/src/models/predictors/Sundial/.venv/bin/python"
    ["TimesFM25"]="${PROJECT_DIR}/src/models/predictors/TimesFM25/.venv/bin/python"
)

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

DRY_RUN=false
FAILED_ONLY=false

REST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        *)            REST_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

is_completed() {
    local exp_name="$1"
    local log_file="${PROJECT_DIR}/logs/${exp_name}.log"
    [[ -f "$log_file" ]] && grep -q "Experiment completed successfully!" "$log_file" 2>/dev/null
}

get_label() {
    case "$1" in
        Supply_Chain_Customer_D) echo "Supply Chain-Customer" ;;
        azure2019_D_5T)          echo "Azure2019-D" ;;
        B1) echo "两阶段基线" ;;
        B2) echo "TIP" ;;
        B3) echo "预训练+TIP" ;;
        *)  echo "$1" ;;
    esac
}

run_one() {
    local exp_type="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"
    local missing_rate="$5"
    local seed="$6"

    local exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${missing_rate}_L${LOOKBACK_LEN}_k${TIP_K}_s${seed}"

    if [[ "$FAILED_ONLY" == true ]] && is_completed "$exp_name"; then
        echo "[SKIP] $exp_name (already completed)"
        return 0
    fi

    local python_bin="${PYTHON_BIN[$predictor]}"
    if [[ ! -x "$python_bin" ]]; then
        echo "[FAIL] $exp_name — Python not found: $python_bin" >&2
        return 1
    fi

    local cmd=(
        "$python_bin" main.py
        --experiment.exp_name               "$exp_name"
        --experiment.exp_type               "$exp_type"
        --experiment.random_seed            "$seed"
        --dataset.dataset                   "$dataset"
        --dataset.missing_rate              "$missing_rate"
        --dataset.artificial_missing_rate   "$missing_rate"
        --dataset.lookback_len              "$LOOKBACK_LEN"
        --imputator.model_type              "$imputator"
        --predictor.model_type              "$predictor"
        --training.tip_k                    "$TIP_K"
        --training.tip_epsilon              "$TIP_EPSILON"
        --training.tip_pretrain_epochs      "$TIP_PRETRAIN_EPOCHS"
    )

    echo "--------------------------------------------------------------------------------"
    echo "[RUN ] $exp_name"
    echo "       $(get_label "$exp_type") | $(get_label "$dataset") | $imputator→$predictor | p=$missing_rate"
    echo "       $(printf '%q ' "${cmd[@]}")"
    echo "--------------------------------------------------------------------------------"

    if [[ "$DRY_RUN" == false ]]; then
        if HF_ENDPOINT="$HF_ENDPOINT" "${cmd[@]}"; then
            echo "[OK  ] $exp_name"
        else
            echo "[FAIL] $exp_name (exit code $?)" >&2
            return 1
        fi
    fi
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

echo "=============================================="
echo "  缺失率鲁棒性实验 — 论文 Section 5.5"
echo "=============================================="
echo "  策略     : ${EXP_TYPES[*]}"
echo "  数据集   : ${DATASETS[*]}"
echo "  插补模型 : ${IMPUTATORS[*]}"
echo "  预测模型 : ${PREDICTORS[*]}"
echo "  缺失率   : ${MISSING_RATES[*]}"
echo "  回顾窗口 : L=$LOOKBACK_LEN"
echo "  TIP-k    : $TIP_K"
echo "  TIP-ε    : $TIP_EPSILON"
echo "  pretrain : ${TIP_PRETRAIN_EPOCHS} epochs"
echo "  dry_run  : $DRY_RUN"
echo "  failed_only: $FAILED_ONLY"
echo "=============================================="
echo ""

total=0
passed=0
failed=0

for exp_type in "${EXP_TYPES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for imputator in "${IMPUTATORS[@]}"; do
            for predictor in "${PREDICTORS[@]}"; do
                for missing_rate in "${MISSING_RATES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        total=$((total + 1))
                        if run_one "$exp_type" "$dataset" "$imputator" "$predictor" "$missing_rate" "$seed"; then
                            passed=$((passed + 1))
                        else
                            failed=$((failed + 1))
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "  缺失率鲁棒性实验汇总"
echo "  Total: $total | Passed: $passed | Failed: $failed"
echo "=============================================="

if [[ $failed -gt 0 ]]; then
    exit 1
fi
