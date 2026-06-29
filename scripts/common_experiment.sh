#!/usr/bin/env bash

# Shared settings and helpers for the paper experiments.
# Source this file from other scripts; do not run it directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
RESULT_DIR="${PROJECT_DIR}/results"

mkdir -p "$LOG_DIR"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PAPER_DATASETS=(
    "Supply_Chain_Customer_D"
    "current_velocity_H"
    "azure2019_D_5T"
    "Coastal_T_S_15T"
    "CPHL_30T"
    "MetroPT-3_5T"
    "Australia_Solar_H"
)

PAPER_IMPUTATORS=(BRITS SAITS DeepMVI TimesNet)
PAPER_PREDICTORS=(Sundial TimesFM25)
PAPER_MISSING_RATES=(0.1 0.2 0.3 0.4 0.5)

DEFAULT_MAIN_MISSING_RATE="0.2"
DEFAULT_LOOKBACK_LEN=128
DEFAULT_BLOCK_SIZE=20
DEFAULT_BATCH_SIZE=128
DEFAULT_MAX_EPOCH=10000
DEFAULT_PATIENCE=5
DEFAULT_DEVICE="cuda"
DEFAULT_SEEDS=(42)
DEFAULT_TIP_K=20
DEFAULT_TIP_EPSILON="1e-4"
DEFAULT_PRETRAIN_EPOCHS=2

declare -A PYTHON_BIN=(
    ["Sundial"]="${PROJECT_DIR}/src/models/predictors/Sundial/.venv/bin/python"
    ["TimesFM25"]="${PROJECT_DIR}/src/models/predictors/TimesFM25/.venv/bin/python"
)

csv_to_array() {
    local target_name="$1"
    local csv="$2"
    local IFS=','
    # shellcheck disable=SC2178
    read -r -a "$target_name" <<< "$csv"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

python_for_predictor() {
    local predictor="$1"
    local python_bin="${PYTHON_BIN[$predictor]:-}"
    if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
        die "Python environment not found for predictor '${predictor}': ${python_bin:-unset}. Run uv sync for that predictor first."
    fi
    echo "$python_bin"
}

eps_label() {
    local epsilon="$1"
    printf '%s' "$epsilon" | tr '+' 'p'
}

experiment_name() {
    local exp_type="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"
    local missing_rate="$5"
    local lookback_len="$6"
    local seed="$7"
    local tip_k="${8:-$DEFAULT_TIP_K}"
    local tip_epsilon="${9:-$DEFAULT_TIP_EPSILON}"
    local pretrain_epochs="${10:-$DEFAULT_PRETRAIN_EPOCHS}"

    if [[ "$exp_type" == "B1" ]]; then
        printf '%s_%s_%s_%s_mr%s_L%s_s%s' \
            "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$lookback_len" "$seed"
    elif [[ "$exp_type" == "B3" ]]; then
        printf '%s_%s_%s_%s_mr%s_L%s_k%s_eps%s_pt%s_s%s' \
            "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$lookback_len" "$tip_k" "$(eps_label "$tip_epsilon")" \
            "$pretrain_epochs" "$seed"
    else
        printf '%s_%s_%s_%s_mr%s_L%s_k%s_eps%s_s%s' \
            "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$lookback_len" "$tip_k" "$(eps_label "$tip_epsilon")" "$seed"
    fi
}

experiment_name_with_suffix() {
    local exp_name
    exp_name="$(experiment_name "$@")"
    if [[ -n "${EXP_NAME_SUFFIX:-}" ]]; then
        exp_name="${exp_name}_${EXP_NAME_SUFFIX}"
    fi
    echo "$exp_name"
}

is_completed() {
    local exp_name="$1"
    local log_file="${LOG_DIR}/${exp_name}.log"
    [[ -f "$log_file" ]] && grep -q "Experiment completed successfully!" "$log_file" 2>/dev/null
}

run_one_experiment() {
    local exp_type="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"
    local missing_rate="$5"
    local seed="$6"
    local lookback_len="${7:-$DEFAULT_LOOKBACK_LEN}"
    local tip_k="${8:-$DEFAULT_TIP_K}"
    local tip_epsilon="${9:-$DEFAULT_TIP_EPSILON}"
    local pretrain_epochs="${10:-$DEFAULT_PRETRAIN_EPOCHS}"
    local dry_run="${11:-false}"
    local failed_only="${12:-false}"
    local max_epoch="${13:-$DEFAULT_MAX_EPOCH}"
    local patience="${14:-$DEFAULT_PATIENCE}"
    local batch_size="${15:-$DEFAULT_BATCH_SIZE}"
    local device="${16:-$DEFAULT_DEVICE}"

    local exp_name
    exp_name="$(experiment_name_with_suffix "$exp_type" "$dataset" "$imputator" "$predictor" \
        "$missing_rate" "$lookback_len" "$seed" "$tip_k" "$tip_epsilon" "$pretrain_epochs")"

    if [[ "$failed_only" == true ]] && is_completed "$exp_name"; then
        echo "[SKIP] $exp_name"
        return 0
    fi

    local python_bin
    python_bin="$(python_for_predictor "$predictor")"

    local cmd=(
        "$python_bin" main.py
        --experiment.exp_name "$exp_name"
        --experiment.exp_type "$exp_type"
        --experiment.random_seed "$seed"
        --dataset.dataset "$dataset"
        --dataset.missing_rate "$missing_rate"
        --dataset.artificial_missing_rate "$missing_rate"
        --dataset.block_size "$DEFAULT_BLOCK_SIZE"
        --dataset.lookback_len "$lookback_len"
        --imputator.model_type "$imputator"
        --predictor.model_type "$predictor"
        --training.batch_size "$batch_size"
        --training.max_epoch "$max_epoch"
        --training.patience "$patience"
        --training.device "$device"
        --training.tip_k "$tip_k"
        --training.tip_epsilon "$tip_epsilon"
        --training.tip_pretrain_epochs "$pretrain_epochs"
    )

    echo "[RUN] $exp_name"
    printf '      '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "$dry_run" == true ]]; then
        return 0
    fi

    (
        cd "$PROJECT_DIR"
        HF_ENDPOINT="$HF_ENDPOINT" "${cmd[@]}"
    )
}

print_common_matrix() {
    local title="$1"
    local exp_types="$2"
    local datasets="$3"
    local imputators="$4"
    local predictors="$5"
    local seeds="$6"
    local missing_rates="$7"
    local lookback_len="$8"
    local tip_k="$9"
    local tip_epsilon="${10}"

    echo "=============================================="
    echo "  $title"
    echo "=============================================="
    echo "  exp types    : $exp_types"
    echo "  datasets     : $datasets"
    echo "  imputators   : $imputators"
    echo "  predictors   : $predictors"
    echo "  missing rates: $missing_rates"
    echo "  lookback L   : $lookback_len"
    echo "  seeds        : $seeds"
    echo "  TIP k        : $tip_k"
    echo "  TIP epsilon  : $tip_epsilon"
    echo "=============================================="
}

job_pool_init() {
    JOB_POOL_JOBS="$1"
    JOB_POOL_DRY_RUN="$2"
    JOB_POOL_FAILED_ONLY="$3"
    JOB_POOL_TOTAL=0
    JOB_POOL_PASSED=0
    JOB_POOL_FAILED=0
    JOB_POOL_SKIPPED=0
    JOB_POOL_PIDS=()
    JOB_POOL_NAMES=()
    JOB_POOL_LOGS=()
    JOB_POOL_EXIT_FILES=()
    JOB_POOL_LOCKS=()
    JOB_POOL_STATUS_DIR="${LOG_DIR}/.runner_status"
    mkdir -p "$JOB_POOL_STATUS_DIR"
    trap 'job_pool_cleanup' EXIT
    trap 'job_pool_cleanup; exit 130' INT
    trap 'job_pool_cleanup; exit 143' TERM
}

job_pool_cleanup() {
    local pid lock_dir
    for pid in "${JOB_POOL_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for lock_dir in "${JOB_POOL_LOCKS[@]:-}"; do
        rmdir "$lock_dir" 2>/dev/null || true
    done
}

job_pool_wait_oldest() {
    if [[ ${#JOB_POOL_PIDS[@]} -eq 0 ]]; then
        return 0
    fi

    local pid="${JOB_POOL_PIDS[0]}"
    local name="${JOB_POOL_NAMES[0]}"
    local runner_log="${JOB_POOL_LOGS[0]}"
    local exit_file="${JOB_POOL_EXIT_FILES[0]}"
    local lock_dir="${JOB_POOL_LOCKS[0]}"
    local code

    set +e
    wait "$pid"
    code=$?
    set -e

    if [[ -f "$exit_file" ]]; then
        code="$(cat "$exit_file")"
        rm -f "$exit_file"
    fi
    rmdir "$lock_dir" 2>/dev/null || true

    if [[ "$code" == "0" ]]; then
        echo "[OK  ] $name"
        JOB_POOL_PASSED=$((JOB_POOL_PASSED + 1))
    elif [[ "$code" == "2" ]]; then
        echo "[SKIP] $name"
        JOB_POOL_SKIPPED=$((JOB_POOL_SKIPPED + 1))
    else
        echo "[FAIL] $name (exit code $code) - runner log: $runner_log" >&2
        JOB_POOL_FAILED=$((JOB_POOL_FAILED + 1))
    fi

    JOB_POOL_PIDS=("${JOB_POOL_PIDS[@]:1}")
    JOB_POOL_NAMES=("${JOB_POOL_NAMES[@]:1}")
    JOB_POOL_LOGS=("${JOB_POOL_LOGS[@]:1}")
    JOB_POOL_EXIT_FILES=("${JOB_POOL_EXIT_FILES[@]:1}")
    JOB_POOL_LOCKS=("${JOB_POOL_LOCKS[@]:1}")
}

job_pool_submit() {
    local exp_type="$1"
    local dataset="$2"
    local imputator="$3"
    local predictor="$4"
    local missing_rate="$5"
    local seed="$6"
    local lookback_len="$7"
    local tip_k="$8"
    local tip_epsilon="$9"
    local pretrain_epochs="${10}"
    local max_epoch="${11}"
    local patience="${12}"
    local batch_size="${13}"
    local device="${14}"
    local exp_name lock_dir runner_log exit_file

    JOB_POOL_TOTAL=$((JOB_POOL_TOTAL + 1))

    exp_name="$(experiment_name_with_suffix "$exp_type" "$dataset" "$imputator" \
        "$predictor" "$missing_rate" "$lookback_len" "$seed" "$tip_k" \
        "$tip_epsilon" "$pretrain_epochs")"

    if [[ "$JOB_POOL_FAILED_ONLY" == true ]] && is_completed "$exp_name"; then
        echo "[SKIP] $exp_name"
        JOB_POOL_SKIPPED=$((JOB_POOL_SKIPPED + 1))
        return 0
    fi

    if [[ "$JOB_POOL_DRY_RUN" == true ]]; then
        if run_one_experiment "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$seed" "$lookback_len" "$tip_k" "$tip_epsilon" \
            "$pretrain_epochs" true "$JOB_POOL_FAILED_ONLY" "$max_epoch" \
            "$patience" "$batch_size" "$device"; then
            JOB_POOL_PASSED=$((JOB_POOL_PASSED + 1))
        else
            JOB_POOL_FAILED=$((JOB_POOL_FAILED + 1))
        fi
        return 0
    fi

    lock_dir="${LOG_DIR}/${exp_name}.runlock"
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "[SKIP] $exp_name (already running; lock: $lock_dir)"
        JOB_POOL_SKIPPED=$((JOB_POOL_SKIPPED + 1))
        return 0
    fi

    if [[ "$JOB_POOL_JOBS" -le 1 ]]; then
        JOB_POOL_LOCKS+=("$lock_dir")
        set +e
        run_one_experiment "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$seed" "$lookback_len" "$tip_k" "$tip_epsilon" \
            "$pretrain_epochs" false false "$max_epoch" "$patience" \
            "$batch_size" "$device"
        local code=$?
        set -e
        rmdir "$lock_dir" 2>/dev/null || true
        JOB_POOL_LOCKS=("${JOB_POOL_LOCKS[@]:1}")
        if [[ "$code" == "0" ]]; then
            JOB_POOL_PASSED=$((JOB_POOL_PASSED + 1))
        else
            JOB_POOL_FAILED=$((JOB_POOL_FAILED + 1))
        fi
        return 0
    fi

    runner_log="${LOG_DIR}/${exp_name}.runner.log"
    exit_file="${JOB_POOL_STATUS_DIR}/${exp_name}.exit"
    rm -f "$exit_file"
    echo "[START] $exp_name - runner log: $runner_log"
    (
        set +e
        run_one_experiment "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$seed" "$lookback_len" "$tip_k" "$tip_epsilon" \
            "$pretrain_epochs" false false "$max_epoch" "$patience" \
            "$batch_size" "$device"
        echo "$?" > "$exit_file"
    ) >"$runner_log" 2>&1 &

    JOB_POOL_PIDS+=("$!")
    JOB_POOL_NAMES+=("$exp_name")
    JOB_POOL_LOGS+=("$runner_log")
    JOB_POOL_EXIT_FILES+=("$exit_file")
    JOB_POOL_LOCKS+=("$lock_dir")

    while [[ ${#JOB_POOL_PIDS[@]} -ge "$JOB_POOL_JOBS" ]]; do
        job_pool_wait_oldest
    done
}

job_pool_wait_all() {
    while [[ ${#JOB_POOL_PIDS[@]} -gt 0 ]]; do
        job_pool_wait_oldest
    done
}

job_pool_print_summary() {
    local title="$1"
    echo ""
    echo "=============================================="
    echo "  $title"
    echo "  Total: $JOB_POOL_TOTAL | Passed: $JOB_POOL_PASSED | Skipped: $JOB_POOL_SKIPPED | Failed: $JOB_POOL_FAILED"
    echo "=============================================="
    if [[ "$JOB_POOL_FAILED" -gt 0 ]]; then
        return 1
    fi
    return 0
}
