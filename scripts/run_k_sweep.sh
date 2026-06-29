#!/usr/bin/env bash

# TIP sampling-count ablation:
#   k in {5, 10, 20, 50, 100}, epsilon fixed at 1e-4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

K_VALUES=(5 10 20 50 100)
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
MISSING_RATE="$DEFAULT_MAIN_MISSING_RATE"
LOOKBACK_LEN="$DEFAULT_LOOKBACK_LEN"
SEEDS=("${DEFAULT_SEEDS[@]}")
MAX_EPOCH="$DEFAULT_MAX_EPOCH"
PATIENCE="$DEFAULT_PATIENCE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
DEVICE="$DEFAULT_DEVICE"
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
DRY_RUN=false
FAILED_ONLY=false
FULL_MATRIX=false
JOBS=1

REP_DATASETS=("Supply_Chain_Customer_D" "azure2019_D_5T")
REP_IMPUTATORS=("SAITS" "DeepMVI")
REP_PREDICTORS=("TimesFM25" "Sundial")

usage() {
    cat <<'EOF'
Usage: scripts/run_k_sweep.sh [options]

Options:
  --k-values 5,10,20,50,100  K values to test.
  --full                     Run all 7 datasets x 4 imputers x 2 predictors.
  --seed 42,43               Random seeds. Default: 42.
  --missing-rate 0.2         Missing rate. Default: 0.2.
  --lookback 512             Lookback length L. Default: 512.
  --epsilon 1e-4             Fixed epsilon. Default: 1e-4.
  --max-epoch N              Max training epochs. Default: 10000.
  --patience N               Early-stopping patience. Default: 5.
  --batch-size N             Effective batch size. Default: 128.
  --device cuda|cpu          Torch device. Default: cuda.
  --jobs N, -j N             Run up to N experiments concurrently. Default: 1.
  --failed-only              Skip completed experiments.
  --dry-run                  Print commands without running them.
  -h, --help                 Show this help.

Default representative configs:
  Supply_Chain_Customer_D + SAITS   + TimesFM25
  azure2019_D_5T           + DeepMVI + Sundial
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k-values) csv_to_array K_VALUES "$2"; shift 2 ;;
        --full) FULL_MATRIX=true; shift ;;
        --seed) csv_to_array SEEDS "$2"; shift 2 ;;
        --missing-rate) MISSING_RATE="$2"; shift 2 ;;
        --lookback) LOOKBACK_LEN="$2"; shift 2 ;;
        --epsilon) TIP_EPSILON="$2"; shift 2 ;;
        --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

if [[ "$FULL_MATRIX" == true ]]; then
    DATASETS=("${PAPER_DATASETS[@]}")
    IMPUTATORS=("${PAPER_IMPUTATORS[@]}")
    PREDICTORS=("${PAPER_PREDICTORS[@]}")
else
    DATASETS=()
    IMPUTATORS=()
    PREDICTORS=()
fi

echo "=============================================="
echo "  K sweep"
echo "=============================================="
echo "  k values     : ${K_VALUES[*]}"
echo "  epsilon      : $TIP_EPSILON"
echo "  missing rate : $MISSING_RATE"
echo "  lookback L   : $LOOKBACK_LEN"
echo "  full matrix  : $FULL_MATRIX"
echo "  seeds        : ${SEEDS[*]}"
echo "  jobs         : $JOBS"
echo "=============================================="
echo ""

job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

run_config() {
    local dataset="$1"
    local imputator="$2"
    local predictor="$3"

    for k in "${K_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_pool_submit "B2" "$dataset" "$imputator" "$predictor" \
                "$MISSING_RATE" "$seed" "$LOOKBACK_LEN" "$k" "$TIP_EPSILON" \
                "$PRETRAIN_EPOCHS" "$MAX_EPOCH" "$PATIENCE" "$BATCH_SIZE" "$DEVICE"
        done
    done
}

if [[ "$FULL_MATRIX" == true ]]; then
    for dataset in "${DATASETS[@]}"; do
        for imputator in "${IMPUTATORS[@]}"; do
            for predictor in "${PREDICTORS[@]}"; do
                run_config "$dataset" "$imputator" "$predictor"
            done
        done
    done
else
    for i in "${!REP_DATASETS[@]}"; do
        run_config "${REP_DATASETS[$i]}" "${REP_IMPUTATORS[$i]}" "${REP_PREDICTORS[$i]}"
    done
fi

job_pool_wait_all
job_pool_print_summary "K sweep summary"
