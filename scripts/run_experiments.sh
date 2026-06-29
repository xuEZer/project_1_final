#!/usr/bin/env bash

# Main paper experiment:
#   B1 two-stage baseline vs B2 TIP on 7 datasets x 4 imputers x 2 TSFMs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

EXP_TYPES=(B1 B2)
DATASETS=("${PAPER_DATASETS[@]}")
IMPUTATORS=("${PAPER_IMPUTATORS[@]}")
PREDICTORS=("${PAPER_PREDICTORS[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
MISSING_RATE="$DEFAULT_MAIN_MISSING_RATE"
LOOKBACK_LEN="$DEFAULT_LOOKBACK_LEN"
TIP_K="$DEFAULT_TIP_K"
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
MAX_EPOCH="$DEFAULT_MAX_EPOCH"
PATIENCE="$DEFAULT_PATIENCE"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
DEVICE="$DEFAULT_DEVICE"
DRY_RUN=false
FAILED_ONLY=false
JOBS=1

usage() {
    cat <<'EOF'
Usage: scripts/run_experiments.sh [options]

Options:
  --exp B1,B2             Experiment types to run. Default: B1,B2.
  --dataset A,B           Dataset config keys. Default: all 7 paper datasets.
  --imputator A,B         Imputers. Default: BRITS,SAITS,DeepMVI,TimesNet.
  --predictor A,B         Predictors. Default: Sundial,TimesFM25.
  --seed 42,43            Random seeds. Default: 42.
  --missing-rate 0.2      Missing rate for main experiment. Default: 0.2.
  --lookback 512          Lookback length L. Default: 512.
  --k 20                  TIP sampling count. Default: 20.
  --epsilon 1e-4          TIP finite-difference step. Default: 1e-4.
  --max-epoch N           Max training epochs. Default: 10000.
  --patience N            Early-stopping patience. Default: 5.
  --batch-size N          Effective batch size. Default: 128.
  --device cuda|cpu       Torch device. Default: cuda.
  --jobs N, -j N          Run up to N experiments concurrently. Default: 1.
  --failed-only           Skip logs that already completed successfully.
  --dry-run               Print commands without running them.
  -h, --help              Show this help.

Examples:
  bash scripts/run_experiments.sh --dry-run
  bash scripts/run_experiments.sh --exp B2 --failed-only --jobs 2
  bash scripts/run_experiments.sh --dataset CPHL_30T --exp B2
  bash scripts/run_experiments.sh --predictor TimesFM25 --imputator SAITS
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
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

print_common_matrix \
    "Main experiment: B1 vs B2" \
    "${EXP_TYPES[*]}" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
    "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "$TIP_K" "$TIP_EPSILON"
echo "  max epoch    : $MAX_EPOCH"
echo "  patience     : $PATIENCE"
echo "  batch size   : $BATCH_SIZE"
echo "  device       : $DEVICE"
echo "  jobs         : $JOBS"
echo "  dry run      : $DRY_RUN"
echo "  failed only  : $FAILED_ONLY"
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
job_pool_print_summary "Main experiment summary"
