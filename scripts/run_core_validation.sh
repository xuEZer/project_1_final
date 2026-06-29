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
LOOKBACK_LEN=128
TIP_K=5
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
MAX_EPOCH=5
PATIENCE=2
BATCH_SIZE=64
DEVICE="$DEFAULT_DEVICE"
JOBS=1
DRY_RUN=false
FAILED_ONLY=false
COLLECT=true

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
  lookback  : 128
  k         : 5
  max epoch : 5
  patience  : 2

Options:
  --exp B1,B2             Experiment types. Default: B1,B2.
  --dataset A,B           Dataset config keys.
  --imputator A,B         Imputers. Default: SAITS,DeepMVI.
  --predictor A,B         Predictor(s). Default: Sundial.
  --seed 42,43            Random seeds. Default: 42.
  --missing-rate 0.2      Missing rate. Default: 0.2.
  --lookback N            Lookback length. Default: 128.
  --k N                   TIP sampling count. Default: 5.
  --epsilon VALUE         TIP finite-difference step. Default: 1e-4.
  --max-epoch N           Max training epochs. Default: 5.
  --patience N            Early-stopping patience. Default: 2.
  --batch-size N          Effective batch size. Default: 64.
  --device cuda|cpu       Torch device. Default: cuda.
  --jobs N, -j N          Concurrent jobs. Default: 1.
  --failed-only           Skip logs already completed successfully.
  --no-collect            Do not run collect_results.py after training.
  --dry-run               Print commands without running them.
  -h, --help              Show this help.

Examples:
  bash scripts/run_core_validation.sh --dry-run
  bash scripts/run_core_validation.sh --failed-only --jobs 1
  bash scripts/run_core_validation.sh --predictor TimesFM25 --batch-size 32
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
        --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --no-collect) COLLECT=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

print_common_matrix \
    "Core validation: B1 vs B2" \
    "${EXP_TYPES[*]}" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
    "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "$TIP_K" "$TIP_EPSILON"
echo "  max epoch    : $MAX_EPOCH"
echo "  patience     : $PATIENCE"
echo "  batch size   : $BATCH_SIZE"
echo "  device       : $DEVICE"
echo "  jobs         : $JOBS"
echo "  dry run      : $DRY_RUN"
echo "  failed only  : $FAILED_ONLY"
echo "  collect      : $COLLECT"
echo ""

job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

for exp_type in "${EXP_TYPES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for imputator in "${IMPUTATORS[@]}"; do
            for predictor in "${PREDICTORS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    job_pool_submit "$exp_type" "$dataset" "$imputator" "$predictor" \
                        "$MISSING_RATE" "$seed" "$LOOKBACK_LEN" "$TIP_K" \
                        "$TIP_EPSILON" "$PRETRAIN_EPOCHS" "$MAX_EPOCH" \
                        "$PATIENCE" "$BATCH_SIZE" "$DEVICE"
                done
            done
        done
    done
done

job_pool_wait_all
SUMMARY_CODE=0
job_pool_print_summary "Core validation summary" || SUMMARY_CODE=$?

if [[ "$COLLECT" == true ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] python3 ${SCRIPT_DIR}/collect_results.py"
    else
        python3 "${SCRIPT_DIR}/collect_results.py"
    fi
fi

exit "$SUMMARY_CODE"
