#!/usr/bin/env bash

# Projection-strategy ablation:
#   B1 baseline, B2 TIP, B4 random direction, B5 unconditional projection,
#   B6 direct prediction-gradient update.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

EXP_TYPES=(B1 B2 B4 B5 B6)
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
Usage: scripts/run_projection_ablation.sh [options]

Options:
  --exp B1,B2,B4,B5,B6     Experiment types. Default: B1,B2,B4,B5,B6.
  --representative         Use a small 2x2x2 representative grid instead of the full matrix.
  --dataset A,B            Dataset config keys. Default: all 7 paper datasets.
  --imputator A,B          Imputers. Default: BRITS,SAITS,DeepMVI,TimesNet.
  --predictor A,B          Predictors. Default: Sundial,TimesFM25.
  --seed 42,43             Random seeds. Default: configs/global.ini.
  --missing-rate 0.2       Missing rate. Default: configs/global.ini.
  --lookback 512           Lookback length L. Default: configs/global.ini.
  --k 20                   TIP sampling count. Default: configs/global.ini.
  --epsilon 1e-4           TIP finite-difference step. Default: configs/global.ini.
  --max-epoch N            Max training epochs. Default: configs/global.ini.
  --patience N             Early-stopping patience. Default: configs/global.ini.
  --batch-size N           Effective batch size. Default: configs/global.ini.
  --device cuda|cpu        Torch device. Default: configs/global.ini.
  --jobs N, -j N           Run up to N experiments concurrently. Default: 1.
  --failed-only            Skip completed experiments.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp) csv_to_array EXP_TYPES "$2"; shift 2 ;;
        --representative)
            DATASETS=("Supply_Chain_Customer_D" "azure2019_D_5T")
            IMPUTATORS=("SAITS" "DeepMVI")
            PREDICTORS=("TimesFM25" "Sundial")
            shift
            ;;
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
    "Projection ablation" \
    "${EXP_TYPES[*]}" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
    "${SEEDS[*]}" "$MISSING_RATE" "$LOOKBACK_LEN" "$TIP_K" "$TIP_EPSILON"
echo "  dry run      : $DRY_RUN"
echo "  failed only  : $FAILED_ONLY"
echo "  jobs         : $JOBS"
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
job_pool_print_summary "Projection ablation summary"
