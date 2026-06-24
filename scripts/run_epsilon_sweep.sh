#!/usr/bin/env bash
#
# TIP 扰动步长 ε 消融实验 — 论文 Section 5.4.2
#
# 用法:
#   ./scripts/run_epsilon_sweep.sh                     # 运行完整 ε-sweep
#   ./scripts/run_epsilon_sweep.sh --dry-run           # 只打印命令
#   ./scripts/run_epsilon_sweep.sh --failed-only       # 只重跑失败的实验
#
# 实验设计:
#   - ε ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}
#   - k = 20 固定
#   - p = 0.2 固定
#   - 代表性配置: Supply Chain-Customer + SAITS + TimesFM25
#                  azure2019_D + DeepMVI + Sundial
#
# 预期 (论文 Section 5.4.2):
#   - ε 过小 (1e-5): 差分信号被数值精度淹没, ĝ_P 退化为噪声
#   - ε 过大 (1e-1): Taylor 近似失效, 扰动超出有效范围
#   - 有效区间: 1e-4 ~ 1e-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------------------------------------------------------------------------
# 实验矩阵
# ---------------------------------------------------------------------------

EPSILON_VALUES=(1e-5 1e-4 1e-3 1e-2 1e-1)
K=20
MISSING_RATE="0.2"
LOOKBACK_LEN=128
SEED=42

# 代表性配置: 覆盖两个不同的 (插补架构 × 预测模型) 组合
declare -A CONFIGS

CONFIGS[0,dataset]="Supply_Chain_Customer_D"
CONFIGS[0,imputator]="SAITS"
CONFIGS[0,predictor]="TimesFM25"

CONFIGS[1,dataset]="azure2019_D_5T"
CONFIGS[1,imputator]="DeepMVI"
CONFIGS[1,predictor]="Sundial"

NUM_CONFIGS=2

declare -A PYTHON_BIN=(
    ["Sundial"]="${PROJECT_DIR}/src/models/predictors/Sundial/.venv/bin/python"
    ["TimesFM25"]="${PROJECT_DIR}/src/models/predictors/TimesFM25/.venv/bin/python"
)

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

DRY_RUN=false
FAILED_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        *)            shift ;;
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

extract_pred_mae() {
    local log_file="$1"
    grep -A 10 "Evaluation Results Summary:" "$log_file" 2>/dev/null \
        | grep "^MAE" \
        | tail -1 \
        | awk '{print $NF}'
}

run_one() {
    local epsilon="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"

    # 用科学计数法格式化 ε 用于实验名
    local eps_str
    eps_str=$(python3 -c "print(f'{float($epsilon):.0e}')" 2>/dev/null || echo "$epsilon")

    local exp_name="B2_${dataset}_${imputator}_${predictor}_mr${MISSING_RATE}_L${LOOKBACK_LEN}_k${K}_eps${eps_str}_s${SEED}"

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
        --experiment.exp_type               "B2"
        --experiment.random_seed            "$SEED"
        --dataset.dataset                   "$dataset"
        --dataset.missing_rate              "$MISSING_RATE"
        --dataset.artificial_missing_rate   "$MISSING_RATE"
        --dataset.lookback_len              "$LOOKBACK_LEN"
        --imputator.model_type              "$imputator"
        --predictor.model_type              "$predictor"
        --training.tip_k                    "$K"
        --training.tip_epsilon              "$epsilon"
    )

    echo "--------------------------------------------------------------------------------"
    echo "[RUN ] ε=$epsilon → $exp_name"
    echo "       $(printf '%q ' "${cmd[@]}")"
    echo "--------------------------------------------------------------------------------"

    if [[ "$DRY_RUN" == false ]]; then
        if HF_ENDPOINT="$HF_ENDPOINT" "${cmd[@]}"; then
            echo "[OK  ] ε=$epsilon → $exp_name"
        else
            echo "[FAIL] ε=$epsilon → $exp_name (exit code $?)" >&2
            return 1
        fi
    fi
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

echo "=============================================="
echo "  ε 消融实验 — 论文 Section 5.4.2"
echo "=============================================="
echo "  ε values  : ${EPSILON_VALUES[*]}"
echo "  k (fixed) : $K"
echo "  mr (fixed): $MISSING_RATE"
echo "  L (fixed) : $LOOKBACK_LEN"
echo "  configs   : $NUM_CONFIGS"
echo "=============================================="
echo ""

total=0
passed=0
failed=0

for ((i=0; i<NUM_CONFIGS; i++)); do
    dataset="${CONFIGS[$i,dataset]}"
    imputator="${CONFIGS[$i,imputator]}"
    predictor="${CONFIGS[$i,predictor]}"

    echo "--- Config $((i+1))/$NUM_CONFIGS: $dataset + $imputator + $predictor ---"
    echo ""

    for epsilon in "${EPSILON_VALUES[@]}"; do
        total=$((total + 1))
        if run_one "$epsilon" "$dataset" "$imputator" "$predictor"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done
    echo ""
done

echo ""
echo "=============================================="
echo "  ε 消融实验汇总"
echo "  Total: $total | Passed: $passed | Failed: $failed"
echo "=============================================="

# 从日志汇总结果
if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "  Pred MAE 汇总:"
    echo "  $(printf '%-10s %-12s %s' 'ε' 'Pred MAE' '实验')"
    echo "  $(printf '%-10s %-12s %s' '----------' '------------' '----')"

    for ((i=0; i<NUM_CONFIGS; i++)); do
        dataset="${CONFIGS[$i,dataset]}"
        imputator="${CONFIGS[$i,imputator]}"
        predictor="${CONFIGS[$i,predictor]}"

        for epsilon in "${EPSILON_VALUES[@]}"; do
            eps_str=$(python3 -c "print(f'{float($epsilon):.0e}')" 2>/dev/null || echo "$epsilon")
            exp_name="B2_${dataset}_${imputator}_${predictor}_mr${MISSING_RATE}_L${LOOKBACK_LEN}_k${K}_eps${eps_str}_s${SEED}"
            log_file="${PROJECT_DIR}/logs/${exp_name}.log"
            if [[ -f "$log_file" ]]; then
                mae=$(extract_pred_mae "$log_file")
                printf "  %-10s %-12s %s\n" "ε=$epsilon" "${mae:-N/A}" "${dataset}/${imputator}/${predictor}"
            fi
        done
        echo ""
    done
fi

if [[ $failed -gt 0 ]]; then
    exit 1
fi
