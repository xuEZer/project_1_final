#!/usr/bin/env bash

# Shared settings and helpers for the paper experiments.
# Source this file from other scripts; do not run it directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
RESULT_DIR="${PROJECT_DIR}/results"
GLOBAL_CONFIG_PATH="${PROJECT_DIR}/configs/global.ini"

mkdir -p "$LOG_DIR"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

config_default() {
    local section="$1"
    local key="$2"
    local value

    if [[ ! -f "$GLOBAL_CONFIG_PATH" ]]; then
        echo "[ERROR] Global config not found: $GLOBAL_CONFIG_PATH" >&2
        return 1
    fi

    value="$(awk -F= -v section="$section" -v key="$key" '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            return s
        }
        /^[[:space:]]*($|[#;])/ { next }
        /^[[:space:]]*\[/ {
            current = $0
            sub(/^[[:space:]]*\[/, "", current)
            sub(/\][[:space:]]*$/, "", current)
            current = trim(current)
            next
        }
        current == section {
            eq = index($0, "=")
            if (eq == 0) next
            name = trim(substr($0, 1, eq - 1))
            if (name == key) {
                value = substr($0, eq + 1)
                sub(/[[:space:]]*[#;].*$/, "", value)
                print trim(value)
                found = 1
                exit
            }
        }
        END { if (!found) exit 1 }
    ' "$GLOBAL_CONFIG_PATH")" || {
        echo "[ERROR] Missing [$section] $key in $GLOBAL_CONFIG_PATH" >&2
        return 1
    }

    printf '%s\n' "$value"
}

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

DEFAULT_EXP_TYPE="$(config_default experiment exp_type)"
DEFAULT_RANDOM_SEED="$(config_default experiment random_seed)"
DEFAULT_DATASET="$(config_default dataset dataset)"
DEFAULT_MAIN_MISSING_RATE="$(config_default dataset missing_rate)"
DEFAULT_ARTIFICIAL_MISSING_RATE="$(config_default dataset artificial_missing_rate)"
DEFAULT_LOOKBACK_LEN="$(config_default dataset lookback_len)"
DEFAULT_BLOCK_SIZE="$(config_default dataset block_size)"
DEFAULT_TIP_K="$(config_default training tip_k)"
DEFAULT_TIP_EPSILON="$(config_default training tip_epsilon)"
DEFAULT_PRETRAIN_EPOCHS="$(config_default training tip_pretrain_epochs)"
DEFAULT_IMPUTATOR="$(config_default imputator model_type)"
DEFAULT_PREDICTOR="$(config_default predictor model_type)"
DEFAULT_SEEDS=("$DEFAULT_RANDOM_SEED")

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

append_config_if_changed() {
    local section="$1"
    local key="$2"
    local value="$3"
    local default_value="$4"

    if [[ -n "$value" && "$value" != "$default_value" ]]; then
        cmd+=(--"${section}.${key}" "$value")
    fi
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
    elif [[ "${INCLUDE_EPS_IN_EXP_NAME:-false}" == true ]]; then
        printf '%s_%s_%s_%s_mr%s_L%s_k%s_eps%s_s%s' \
            "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$lookback_len" "$tip_k" "$(eps_label "$tip_epsilon")" "$seed"
    else
        printf '%s_%s_%s_%s_mr%s_L%s_k%s_s%s' \
            "$exp_type" "$dataset" "$imputator" "$predictor" \
            "$missing_rate" "$lookback_len" "$tip_k" "$seed"
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
    local extra_config_args=("${@:13}")

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
    )
    append_config_if_changed experiment exp_type "$exp_type" "$DEFAULT_EXP_TYPE"
    append_config_if_changed experiment random_seed "$seed" "$DEFAULT_RANDOM_SEED"
    append_config_if_changed dataset dataset "$dataset" "$DEFAULT_DATASET"
    append_config_if_changed dataset missing_rate "$missing_rate" "$DEFAULT_MAIN_MISSING_RATE"
    append_config_if_changed dataset artificial_missing_rate "$missing_rate" "$DEFAULT_ARTIFICIAL_MISSING_RATE"
    append_config_if_changed dataset lookback_len "$lookback_len" "$DEFAULT_LOOKBACK_LEN"
    append_config_if_changed imputator model_type "$imputator" "$DEFAULT_IMPUTATOR"
    append_config_if_changed predictor model_type "$predictor" "$DEFAULT_PREDICTOR"
    case "$exp_type" in
        B2|B3|B5|B6)
            append_config_if_changed training tip_k "$tip_k" "$DEFAULT_TIP_K"
            append_config_if_changed training tip_epsilon "$tip_epsilon" "$DEFAULT_TIP_EPSILON"
            ;;
    esac
    if [[ "$exp_type" == "B3" ]]; then
        append_config_if_changed training tip_pretrain_epochs "$pretrain_epochs" "$DEFAULT_PRETRAIN_EPOCHS"
    fi
    if [[ ${#extra_config_args[@]} -gt 0 ]]; then
        cmd+=("${extra_config_args[@]}")
    fi

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

job_pool_reap_at() {
    local idx="$1"
    local already_waited="${2:-false}"
    if [[ ${#JOB_POOL_PIDS[@]} -eq 0 ]]; then
        return 0
    fi

    local pid="${JOB_POOL_PIDS[$idx]}"
    local name="${JOB_POOL_NAMES[$idx]}"
    local runner_log="${JOB_POOL_LOGS[$idx]}"
    local exit_file="${JOB_POOL_EXIT_FILES[$idx]}"
    local lock_dir="${JOB_POOL_LOCKS[$idx]}"
    local code

    if [[ "$already_waited" == true ]]; then
        code=0
    else
        set +e
        wait "$pid"
        code=$?
        set -e
    fi

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

    unset 'JOB_POOL_PIDS[idx]'
    unset 'JOB_POOL_NAMES[idx]'
    unset 'JOB_POOL_LOGS[idx]'
    unset 'JOB_POOL_EXIT_FILES[idx]'
    unset 'JOB_POOL_LOCKS[idx]'
    JOB_POOL_PIDS=("${JOB_POOL_PIDS[@]}")
    JOB_POOL_NAMES=("${JOB_POOL_NAMES[@]}")
    JOB_POOL_LOGS=("${JOB_POOL_LOGS[@]}")
    JOB_POOL_EXIT_FILES=("${JOB_POOL_EXIT_FILES[@]}")
    JOB_POOL_LOCKS=("${JOB_POOL_LOCKS[@]}")
}

job_pool_wait_any() {
    if [[ ${#JOB_POOL_PIDS[@]} -eq 0 ]]; then
        return 0
    fi

    local finished_pid=""
    local idx

    set +e
    wait -n -p finished_pid "${JOB_POOL_PIDS[@]}"
    set -e

    for idx in "${!JOB_POOL_PIDS[@]}"; do
        if [[ "${JOB_POOL_PIDS[$idx]}" == "$finished_pid" ]]; then
            job_pool_reap_at "$idx" true
            return 0
        fi
    done

    # Fallback for shells that return an empty/non-matching -p value.
    job_pool_reap_at 0
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
            "$pretrain_epochs" true "$JOB_POOL_FAILED_ONLY"; then
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
            "$pretrain_epochs" false false
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
            "$pretrain_epochs" false false
        echo "$?" > "$exit_file"
    ) >"$runner_log" 2>&1 &

    JOB_POOL_PIDS+=("$!")
    JOB_POOL_NAMES+=("$exp_name")
    JOB_POOL_LOGS+=("$runner_log")
    JOB_POOL_EXIT_FILES+=("$exit_file")
    JOB_POOL_LOCKS+=("$lock_dir")

    while [[ ${#JOB_POOL_PIDS[@]} -ge "$JOB_POOL_JOBS" ]]; do
        job_pool_wait_any
    done
}

job_pool_wait_all() {
    while [[ ${#JOB_POOL_PIDS[@]} -gt 0 ]]; do
        job_pool_wait_any
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
