#!/usr/bin/env bash

# First-layer core validation:
#   Verify whether TIP (B2) improves over the two-stage baseline (B1)
#   under a small, affordable matrix before launching full paper runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

EXP_TYPES=(B1 B2)
DATASETS=(
    "Supply_Chain_Customer_D"
    "current_velocity_H"
    "azure2019_D_5T"
)
IMPUTATORS=(SAITS DeepMVI)
PREDICTORS=(Sundial)
SEEDS=("${DEFAULT_SEEDS[@]}")
MISSING_RATE="$DEFAULT_MAIN_MISSING_RATE"
LOOKBACK_LEN="$DEFAULT_LOOKBACK_LEN"
TIP_K="$DEFAULT_TIP_K"
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
K_VALUES=(5 10 20)
EPSILON_VALUES=(1e-5 1e-4 1e-3)
EPSILON_K=""
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
JOBS=1
DRY_RUN=false
FAILED_ONLY=false
COLLECT=true
RUN_CORE=true
WITH_SENSITIVITY=false

usage() {
    cat <<'EOF'
Usage: scripts/run_core_validation.sh [options]

Purpose:
  Small first-layer experiment for validating the core conclusion:
  B2/TIP vs B1 on a representative, affordable matrix.

Default matrix:
  exp       : B1,B2
  datasets  : Supply_Chain_Customer_D,current_velocity_H,azure2019_D_5T
  imputers  : SAITS,DeepMVI
  predictor : Sundial
  lookback  : configs/global.ini
  k         : configs/global.ini

Optional sensitivity phases:
  k sweep       : B2 with k in 5,10,20 and epsilon fixed.
  epsilon sweep : B2 with epsilon in 1e-5,1e-4,1e-3 and k fixed by --epsilon-k.

Options:
  --exp B1,B2             Experiment types. Default: B1,B2.
  --dataset A,B           Dataset config keys.
  --imputator A,B         Imputers. Default: SAITS,DeepMVI.
  --predictor A,B         Predictor(s). Default: Sundial.
  --seed 42,43            Random seeds. Default: configs/global.ini.
  --missing-rate 0.2      Missing rate. Default: configs/global.ini.
  --lookback N            Lookback length. Default: configs/global.ini.
  --k N                   TIP sampling count. Default: configs/global.ini.
  --epsilon VALUE         TIP finite-difference step. Default: configs/global.ini.
  --with-sensitivity      Also run the small B2 k/epsilon sensitivity probes.
  --sensitivity-only      Run only the small B2 k/epsilon sensitivity probes.
  --k-values 5,10,20      K values for sensitivity. Default: 5,10,20.
  --epsilon-values LIST   Epsilon values for sensitivity. Default: 1e-5,1e-4,1e-3.
  --epsilon-k N           Fixed k for epsilon sensitivity. Default: same as --k.
  --jobs N, -j N          Concurrent jobs. Default: 1.
  --failed-only           Skip logs already completed successfully.
  --no-collect            Do not run collect_results.py after training.
  --dry-run               Print commands without running them.
  -h, --help              Show this help.

Examples:
  bash scripts/run_core_validation.sh --dry-run
  bash scripts/run_core_validation.sh --failed-only --jobs 1
  bash scripts/run_core_validation.sh --with-sensitivity --dry-run
  bash scripts/run_core_validation.sh --sensitivity-only --epsilon-k 10
  bash scripts/run_core_validation.sh --predictor Sundial
  bash scripts/run_core_validation.sh --dataset CPHL_30T,azure2019_D_5T
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp) csv_to_array EXP_TYPES "$2"; shift 2 ;;
        --dataset) csv_to_array DATASETS "$2"; shift 2 ;;
        --imputator) csv_to_array IMPUTATORS "$2"; shift 2 ;;
        --predictor) csv_to_array PREDICTORS "$2"; shift 2 ;;
        --seed) csv_to_array SEEDS "$2"; shift 2 ;;
        --missing-rate) MISSING_RATE="$2"; shift 2 ;;
        --lookback) LOOKBACK_LEN="$2"; shift 2 ;;
        --k) TIP_K="$2"; shift 2 ;;
        --epsilon) TIP_EPSILON="$2"; shift 2 ;;
        --with-sensitivity) WITH_SENSITIVITY=true; shift ;;
        --sensitivity-only) RUN_CORE=false; WITH_SENSITIVITY=true; shift ;;
        --k-values) csv_to_array K_VALUES "$2"; shift 2 ;;
        --epsilon-values) csv_to_array EPSILON_VALUES "$2"; shift 2 ;;
        --epsilon-k) EPSILON_K="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --no-collect) COLLECT=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

if [[ -z "$EPSILON_K" ]]; then
    EPSILON_K="$TIP_K"
fi

cd "$PROJECT_DIR"

SUMMARY_CODE=0

print_run_settings() {
    echo "  jobs         : $JOBS"
    echo "  dry run      : $DRY_RUN"
    echo "  failed only  : $FAILED_ONLY"
    echo "  collect      : $COLLECT"
    echo ""
}

run_core_phase() {
    print_common_matrix \
        "Core validation: B1 vs B2" \
        "${EXP_TYPES[*]}" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
        "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "$TIP_K" "$TIP_EPSILON"
    print_run_settings

    job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

    for exp_type in "${EXP_TYPES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        job_pool_submit "$exp_type" "$dataset" "$imputator" "$predictor" \
                            "$MISSING_RATE" "$seed" "$LOOKBACK_LEN" "$TIP_K" \
                            "$TIP_EPSILON" "$PRETRAIN_EPOCHS"
                    done
                done
            done
        done
    done

    job_pool_wait_all
    job_pool_print_summary "Core validation summary" || SUMMARY_CODE=$?
}

run_k_sensitivity_phase() {
    print_common_matrix \
        "Core sensitivity: B2 k sweep" \
        "B2" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
        "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "${K_VALUES[*]}" "$TIP_EPSILON"
    print_run_settings

    job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

    for k in "${K_VALUES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        job_pool_submit "B2" "$dataset" "$imputator" "$predictor" \
                            "$MISSING_RATE" "$seed" "$LOOKBACK_LEN" "$k" \
                            "$TIP_EPSILON" "$PRETRAIN_EPOCHS"
                    done
                done
            done
        done
    done

    job_pool_wait_all
    job_pool_print_summary "Core sensitivity k summary" || SUMMARY_CODE=$?
}

run_epsilon_sensitivity_phase() {
    print_common_matrix \
        "Core sensitivity: B2 epsilon sweep" \
        "B2" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
        "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "$EPSILON_K" "${EPSILON_VALUES[*]}"
    print_run_settings

    job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

    for epsilon in "${EPSILON_VALUES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for imputator in "${IMPUTATORS[@]}"; do
                for predictor in "${PREDICTORS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        job_pool_submit "B2" "$dataset" "$imputator" "$predictor" \
                            "$MISSING_RATE" "$seed" "$LOOKBACK_LEN" "$EPSILON_K" \
                            "$epsilon" "$PRETRAIN_EPOCHS"
                    done
                done
            done
        done
    done

    job_pool_wait_all
    job_pool_print_summary "Core sensitivity epsilon summary" || SUMMARY_CODE=$?
}

if [[ "$RUN_CORE" == true ]]; then
    run_core_phase
fi

if [[ "$WITH_SENSITIVITY" == true ]]; then
    run_k_sensitivity_phase
    run_epsilon_sensitivity_phase
fi

if [[ "$COLLECT" == true ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] python3 ${SCRIPT_DIR}/collect_results.py"
    else
        python3 "${SCRIPT_DIR}/collect_results.py"
    fi
fi

exit "$SUMMARY_CODE"
