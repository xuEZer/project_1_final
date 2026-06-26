#!/usr/bin/env bash
#
# 主实验脚本 — 论文 Section 5.3 预测性能对比 (RQ1)
#
# 用法:
#   ./scripts/run_experiments.sh                           # 运行完整实验矩阵
#   ./scripts/run_experiments.sh --dry-run                 # 只打印命令,不执行
#   ./scripts/run_experiments.sh --failed-only             # 只重跑失败的实验
#   ./scripts/run_experiments.sh --jobs 4                  # 同时运行 4 个实验
#   ./scripts/run_experiments.sh -j 4 --failed-only        # 4 并行 + 只重跑失败
#   ./scripts/run_experiments.sh --exp B2                  # 只运行 B2 (TIP)
#   ./scripts/run_experiments.sh --exp B1,B2               # 只运行 B1 和 B2
#   ./scripts/run_experiments.sh --dataset Supply_Chain_Customer_D  # 指定单个数据集
#
# 实验矩阵:
#   - 策略: B1 (两阶段基线), B2 (TIP). B4/B5 消融通过 --exp B4,B5 单独运行
#   - 数据集: 7 个真实世界数据集 (覆盖供应链、物理、云计算、环境、公共卫生、运维、能源)
#   - 插补模型: SAITS, BRITS, DeepMVI, TimesNet
#   - 预测模型: Sundial, TimesFM25
#   - 固定参数: 缺失率 p=0.2, block_size=20, 回顾窗口 L=128
#
#   B3 (预训练+TIP) 缺失率鲁棒性实验见 scripts/run_missing_rate_robustness.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p logs

# HuggingFace 镜像 (国内加速)
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------------------------------------------------------------------------
# 默认参数矩阵 — 按论文 Section 5.3 配置
# ---------------------------------------------------------------------------

# 论文中 7 个真实世界数据集 (对应 configs/template/dataset/ 下的配置)
DATASETS=(
    "Supply_Chain_Customer_D"    # 供应链
    "current_velocity_H"         # 物理测量
    "azure2019_D_5T"             # 工业云计算
    "Coastal_T_S_15T"            # 环境监测
    "CPHL_30T"                   # 公共卫生
    "MetroPT-3_5T"               # 工业运维
    "Australia_Solar_H"          # 能源气象
)

# 4 种架构各异的插补模型
IMPUTATORS=(SAITS BRITS DeepMVI TimesNet)

# 2 种 Decoder-only TSFM
PREDICTORS=(Sundial TimesFM25)

# 主实验固定缺失率 p=0.2 (论文 Section 5.2)
MAIN_MISSING_RATE="0.2"

# 固定回顾窗口 L=512 (论文 Section 5.2)
LOOKBACK_LEN=512

# 固定随机种子
SEEDS=(42)

# TIP 超参数 (论文 Section 4.5)
TIP_K=20
TIP_EPSILON="1e-4"

# 各 predictor 对应的 Python 环境 (与 Makefile 保持一致)
declare -A PYTHON_BIN=(
    ["Sundial"]="${PROJECT_DIR}/src/models/predictors/Sundial/.venv/bin/python"
    ["TimesFM25"]="${PROJECT_DIR}/src/models/predictors/TimesFM25/.venv/bin/python"
)

# ---------------------------------------------------------------------------
# 默认标志
# ---------------------------------------------------------------------------

JOBS=1
DRY_RUN=false
FAILED_ONLY=false
SELECTED_EXP_TYPES=()
SELECTED_DATASETS=()

# ---------------------------------------------------------------------------
# 解析参数
# ---------------------------------------------------------------------------

REST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --failed-only)
            FAILED_ONLY=true
            shift
            ;;
        --exp)
            IFS=',' read -r -a SELECTED_EXP_TYPES <<< "$2"
            shift 2
            ;;
        --dataset)
            SELECTED_DATASETS+=("$2")
            shift 2
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        *)
            REST_ARGS+=("$1")
            shift
            ;;
    esac
done

# 若未指定实验类型,默认运行主实验策略 (B1, B2)。
# B4/B5 消融可通过 --exp B4,B5 单独运行。
# B3 在缺失率鲁棒性实验中单独评估 (见 run_missing_rate_robustness.sh)。
if [[ ${#SELECTED_EXP_TYPES[@]} -eq 0 ]]; then
    SELECTED_EXP_TYPES=(B1 B2)
fi

# 若未指定数据集,使用全部 7 个
if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    SELECTED_DATASETS=("${DATASETS[@]}")
fi

# ---------------------------------------------------------------------------
# 辅助函数
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

get_exp_type_label() {
    case "$1" in
        B1) echo "两阶段基线" ;;
        B2) echo "TIP" ;;
        B4) echo "随机方向消融" ;;
        B5) echo "无条件投影消融" ;;
        *)   echo "$1" ;;
        
    esac
}

get_dataset_label() {
    case "$1" in
        Supply_Chain_Customer_D) echo "Supply Chain-Customer" ;;
        current_velocity_H)      echo "Current Velocity" ;;
        azure2019_D_5T)          echo "Azure2019-D" ;;
        Coastal_T_S_15T)         echo "Coastal T-S" ;;
        CPHL_30T)                echo "CPHL" ;;
        MetroPT-3_5T)            echo "MetroPT-3" ;;
        Australia_Solar_H)       echo "Australia Solar" ;;
        *)                       echo "$1" ;;
    esac
}

# ---------------------------------------------------------------------------
# 运行单个实验
# ---------------------------------------------------------------------------

run_one() {
    local exp_type="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"
    local missing_rate="$5"
    local seed="$6"

    local exp_label
    exp_label="$(get_exp_type_label "$exp_type")"

    # 构造实验名称: 对非 B1 策略记录 k 值 (B2/B4/B5)
    local exp_name
    if [[ "$exp_type" == "B1" ]]; then
        exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${missing_rate}_L${LOOKBACK_LEN}_s${seed}"
    else
        exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${missing_rate}_L${LOOKBACK_LEN}_k${TIP_K}_s${seed}"
    fi

    if [[ "$FAILED_ONLY" == true ]] && is_completed "$exp_name"; then
        echo "[SKIP] $exp_name (already completed)"
        return 0
    fi

    local python_bin="${PYTHON_BIN[$predictor]:-${PYTHON_BIN[Sundial]}}"

    if [[ ! -x "$python_bin" ]]; then
        echo "[FAIL] $exp_name — Python not found: $python_bin" >&2
        return 1
    fi

    # 构建 CLI 参数
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
    )

    echo "--------------------------------------------------------------------------------"
    echo "[RUN ] $exp_name"
    echo "       策略: $exp_label | 数据集: $(get_dataset_label "$dataset")"
    echo "       插补: $imputator | 预测: $predictor | mr=$missing_rate | L=$LOOKBACK_LEN"
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
# 并行执行辅助函数 (job pool)
# ---------------------------------------------------------------------------

declare -A _PID_TO_NAME _PID_TO_EXIT

_reap_finished() {
    # 回收所有已完成的 background jobs,打印结果并更新计数
    local new_pids=()
    for p in "${_POOL_PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
            new_pids+=("$p")
        else
            local name="${_PID_TO_NAME[$p]}"
            local exit_file="${_PID_TO_EXIT[$p]}"
            if [[ -f "$exit_file" ]]; then
                local code
                code="$(cat "$exit_file")"
                if [[ "$code" == "0" ]]; then
                    echo "[OK  ] $name"
                    passed=$((passed + 1))
                else
                    echo "[FAIL] $name (exit code $code)"
                    failed=$((failed + 1))
                fi
                rm -f "$exit_file"
            else
                echo "[FAIL] $name (no exit code)"
                failed=$((failed + 1))
            fi
            unset '_PID_TO_NAME[$p]'
            unset '_PID_TO_EXIT[$p]'
        fi
    done
    _POOL_PIDS=("${new_pids[@]}")
}

_launch_one() {
    local exp_type="$1" dataset="$2" imputator="$3" predictor="$4"
    local missing_rate="$5" seed="$6"

    local exp_name
    if [[ "$exp_type" == "B1" ]]; then
        exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${missing_rate}_L${LOOKBACK_LEN}_s${seed}"
    else
        exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${missing_rate}_L${LOOKBACK_LEN}_k${TIP_K}_s${seed}"
    fi

    if [[ "$FAILED_ONLY" == true ]] && is_completed "$exp_name"; then
        skipped=$((skipped + 1))
        return 0
    fi

    local log_file="${PROJECT_DIR}/logs/${exp_name}.log"
    local exit_file="${PROJECT_DIR}/logs/${exp_name}.exit"
    rm -f "$exit_file"

    echo "[START] $exp_name  | 策略: $(get_exp_type_label "$exp_type")"
    echo "        数据集: $(get_dataset_label "$dataset") | 插补: $imputator | 预测: $predictor | mr=$missing_rate"
    echo "        log: $log_file"

    (
        if run_one "$exp_type" "$dataset" "$imputator" "$predictor" "$missing_rate" "$seed"; then
            echo "0" > "$exit_file"
        else
            echo "$?" > "$exit_file"
        fi
    ) > "$log_file" 2>&1 &

    local pid=$!
    _POOL_PIDS+=("$pid")
    _PID_TO_NAME[$pid]="$exp_name"
    _PID_TO_EXIT[$pid]="$exit_file"
    total=$((total + 1))
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

echo "=============================================="
echo "  主实验 — 论文 Section 5.3"
echo "=============================================="
echo "  策略     : ${SELECTED_EXP_TYPES[*]}"
echo "  数据集   : ${SELECTED_DATASETS[*]}"
echo "  插补模型 : ${IMPUTATORS[*]}"
echo "  预测模型 : ${PREDICTORS[*]}"
echo "  缺失率   : $MAIN_MISSING_RATE"
echo "  回顾窗口 : L=$LOOKBACK_LEN"
echo "  TIP-k    : $TIP_K"
echo "  TIP-ε    : $TIP_EPSILON"
echo "  seed     : ${SEEDS[*]}"
echo "  jobs     : $JOBS"
echo "  dry_run  : $DRY_RUN"
echo "  failed_only: $FAILED_ONLY"
echo "=============================================="
echo ""

total=0
passed=0
failed=0
skipped=0
_POOL_PIDS=()

if [[ $JOBS -gt 1 ]]; then
    # =========================================================================
    # 并行模式 — job pool,最多 JOBS 个实验同时运行
    # =========================================================================
    echo "  并行模式: 最多 $JOBS 个实验同时运行"
    echo ""

    for exp_type in "${SELECTED_EXP_TYPES[@]}"; do
        for dataset in "${SELECTED_DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        if [[ "$DRY_RUN" == true ]]; then
                            run_one "$exp_type" "$dataset" "$imputator" "$predictor" "$MAIN_MISSING_RATE" "$seed" || true
                            continue
                        fi

                        _launch_one "$exp_type" "$dataset" "$imputator" "$predictor" "$MAIN_MISSING_RATE" "$seed" || continue

                        # 池满则回收已完成的任务
                        while [[ ${#_POOL_PIDS[@]} -ge $JOBS ]]; do
                            sleep 1
                            _reap_finished
                        done
                    done
                done
            done
        done
    done

    # 等待剩余任务全部完成
    echo ""
    echo "  等待剩余 ${#_POOL_PIDS[@]} 个实验完成..."
    while [[ ${#_POOL_PIDS[@]} -gt 0 ]]; do
        sleep 1
        _reap_finished
    done

else
    # =========================================================================
    # 串行模式 — 保持原有行为
    # =========================================================================

    for exp_type in "${SELECTED_EXP_TYPES[@]}"; do
        for dataset in "${SELECTED_DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        exp_name="${exp_type}_${dataset}_${imputator}_${predictor}_mr${MAIN_MISSING_RATE}_L${LOOKBACK_LEN}"
                        if [[ "$exp_type" != "B1" ]]; then
                            exp_name="${exp_name}_k${TIP_K}"
                        fi
                        exp_name="${exp_name}_s${seed}"

                        if [[ "$FAILED_ONLY" == true ]] && is_completed "$exp_name"; then
                            skipped=$((skipped + 1))
                            continue
                        fi
                        total=$((total + 1))
                    done
                done
            done
        done
    done

    if [[ "$FAILED_ONLY" == true ]]; then
        echo "  已完成 (跳过): $skipped"
    fi
    echo "  待运行: $total"
    echo ""

    for exp_type in "${SELECTED_EXP_TYPES[@]}"; do
        for dataset in "${SELECTED_DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        if run_one "$exp_type" "$dataset" "$imputator" "$predictor" "$MAIN_MISSING_RATE" "$seed"; then
                            passed=$((passed + 1))
                        else
                            failed=$((failed + 1))
                        fi
                    done
                done
            done
        done
    done
fi

echo ""
echo "=============================================="
echo "  主实验汇总"
echo "  Total: $total | Passed: $passed | Failed: $failed"
echo "=============================================="

if [[ $failed -gt 0 ]]; then
    exit 1
fi
