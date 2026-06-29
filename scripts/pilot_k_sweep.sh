#!/usr/bin/env bash
#
# 快速 Pilot: 在1个代表性配置上粗扫 k ∈ {5, 10, 20, 50}
#
# 目标: 确认 k=20 是否落在性能-代价的合理折衷区间
#
# 用法:
#   ./scripts/pilot_k_sweep.sh                        # 使用默认配置
#   ./scripts/pilot_k_sweep.sh --dry-run              # 只打印命令,不执行
#   ./scripts/pilot_k_sweep.sh --skip-completed       # 跳过已完成的实验
#
# 可修改下面的 REPRESENTATIVE_CONFIG 来切换配置

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 代表性配置 (挑一个即可,选你手头跑得最快的)
# ---------------------------------------------------------------------------

DATASET="Supply_Chain_Customer_D"
IMPUTATOR="SAITS"
PREDICTOR="TimesFM25"       # TSFM
EXP_TYPE="B2"               # 主 TIP 实验
MISSING_RATE="0.1"
SEED="${DEFAULT_SEEDS[0]}"

K_VALUES=(5 10 20)

# TimesFM25 环境 (有 torch + timesfm 依赖)
PYTHON_BIN="$(python_for_predictor "$PREDICTOR")"

# HuggingFace 镜像 (国内加速)
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DRY_RUN=false
SKIP_COMPLETED=false

# ---------------------------------------------------------------------------
# 解析参数
# ---------------------------------------------------------------------------

REST_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            ;;
        *)
            REST_ARGS+=("$arg")
            ;;
    esac
done
set -- "${REST_ARGS[@]}"

# ---------------------------------------------------------------------------
# 判断实验是否已完成
# ---------------------------------------------------------------------------

is_completed() {
    local exp_name="$1"
    local log_file="${PROJECT_DIR}/logs/${exp_name}.log"
    if [[ -f "$log_file" ]]; then
        grep -q "Experiment completed successfully!" "$log_file" 2>/dev/null
    else
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 从日志中提取 pred MAE
# ---------------------------------------------------------------------------

extract_pred_mae() {
    local log_file="$1"
    # 取最后一组 Evaluation Results 中的 pred MAE (最后一列)
    grep -A 10 "Evaluation Results Summary:" "$log_file" \
        | grep "MAE" \
        | tail -1 \
        | awk '{print $NF}'
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

echo "=============================================="
echo "  Pilot: k-sweep on representative config"
echo "  dataset   : $DATASET"
echo "  imputator : $IMPUTATOR"
echo "  predictor : $PREDICTOR"
echo "  exp_type  : $EXP_TYPE"
echo "  m_rate    : $MISSING_RATE"
echo "  k values  : ${K_VALUES[*]}"
echo "  hf_mirror : $HF_ENDPOINT"
echo "  dry_run   : $DRY_RUN"
echo "=============================================="
echo ""

declare -A RESULTS

for k in "${K_VALUES[@]}"; do
    exp_name="${EXP_TYPE}_${DATASET}_${IMPUTATOR}_${PREDICTOR}_mr${MISSING_RATE}_k${k}_s${SEED}"

    if [[ "$SKIP_COMPLETED" == true ]] && is_completed "$exp_name"; then
        echo "[SKIP] $exp_name (already completed)"
        mae=$(extract_pred_mae "${PROJECT_DIR}/logs/${exp_name}.log")
        RESULTS[$k]="$mae"
        continue
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --experiment.exp_name      "$exp_name"
    )
    append_config_if_changed experiment exp_type "$EXP_TYPE" "$DEFAULT_EXP_TYPE"
    append_config_if_changed experiment random_seed "$SEED" "$DEFAULT_RANDOM_SEED"
    append_config_if_changed dataset dataset "$DATASET" "$DEFAULT_DATASET"
    append_config_if_changed dataset missing_rate "$MISSING_RATE" "$DEFAULT_MAIN_MISSING_RATE"
    append_config_if_changed dataset artificial_missing_rate "$MISSING_RATE" "$DEFAULT_ARTIFICIAL_MISSING_RATE"
    append_config_if_changed imputator model_type "$IMPUTATOR" "$DEFAULT_IMPUTATOR"
    append_config_if_changed predictor model_type "$PREDICTOR" "$DEFAULT_PREDICTOR"
    append_config_if_changed training tip_k "$k" "$DEFAULT_TIP_K"

    echo "--------------------------------------------------------------------------------"
    echo "[RUN ] k=$k → $exp_name"
    echo "       ${cmd[*]}"
    echo "--------------------------------------------------------------------------------"

    if [[ "$DRY_RUN" == false ]]; then
        if HF_ENDPOINT="$HF_ENDPOINT" "${cmd[@]}"; then
            echo "[OK  ] k=$k → $exp_name"
        else
            echo "[FAIL] k=$k → $exp_name (exit code $?)" >&2
            RESULTS[$k]="FAILED"
            continue
        fi
    fi

    log_file="${PROJECT_DIR}/logs/${exp_name}.log"
    if [[ -f "$log_file" ]]; then
        mae=$(extract_pred_mae "$log_file")
        RESULTS[$k]="$mae"
        echo "       pred_MAE = $mae"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "Dry-run complete. 以上命令可供检查."
    exit 0
fi

echo ""
echo "=============================================="
echo "  Pilot 结果汇总: k vs Pred MAE"
echo "=============================================="
echo ""
printf "  %-8s %-20s %s\n" "k" "Pred MAE" "备注"
printf "  %-8s %-20s %s\n" "--------" "--------------------" "----"

# 找到基准 MAE (最小的 k) 用于比较
BASELINE_MAE=""
for k in "${K_VALUES[@]}"; do
    if [[ -n "${RESULTS[$k]:-}" ]] && [[ "${RESULTS[$k]}" != "FAILED" ]]; then
        BASELINE_MAE="${RESULTS[$k]}"
        break
    fi
done

for k in "${K_VALUES[@]}"; do
    mae="${RESULTS[$k]:-N/A}"
    note=""
    if [[ "$mae" == "FAILED" ]]; then
        note="❌ 运行失败"
    elif [[ "$mae" == "N/A" ]]; then
        note="未运行"
    elif [[ -n "$BASELINE_MAE" ]] && [[ "$mae" != "$BASELINE_MAE" ]]; then
        # 计算相对基准的变化百分比 (使用 bc 做浮点运算)
        delta=$(python3 -c "print(f'{(float($mae) - float($BASELINE_MAE)) / float($BASELINE_MAE) * 100:+.1f}%')" 2>/dev/null || echo "?")
        note="vs k=min: $delta"
    fi
    printf "  %-8s %-20s %s\n" "k=$k" "$mae" "$note"
done

echo ""
echo "=============================================="
echo "  诊断建议 (根据结果判断):"
echo "=============================================="
echo "  [ ] k=10 vs k=20 差异在噪声范围 → 直接用 k=10 (省一半开销)"
echo "  [ ] k=20 比 k=10 明显好, k=50 基本饱和 → 确认 k=20"
echo "  [ ] 所有 k 差异都很小 → 重新审视论文叙事"
echo "=============================================="
