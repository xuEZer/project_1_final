#!/usr/bin/env bash

# Missing-rate robustness and B3 pretrain+TIP experiment:
#   datasets: Supply_Chain_Customer_D, azure2019_D_5T
#   imputers: SAITS, DeepMVI
#   predictors: Chronos, Timer, Sundial
#   missing rates: 0.1, 0.2, 0.3, 0.4, 0.5
#   exp types: B1, B2, B3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

EXP_TYPES=(B1 B2 B3)
DATASETS=("Supply_Chain_Customer_D" "azure2019_D_5T")
IMPUTATORS=(SAITS DeepMVI)
PREDICTORS=("${PAPER_PREDICTORS[@]}")
MISSING_RATES=("${PAPER_MISSING_RATES[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
LOOKBACK_LEN="$DEFAULT_LOOKBACK_LEN"
TIP_K="$DEFAULT_TIP_K"
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
DRY_RUN=false
FAILED_ONLY=false
JOBS=1

usage() {
    cat <<'EOF'
Usage: scripts/run_missing_rate_robustness.sh [options]

Options:
  --exp B1,B2,B3           Experiment types. Default: B1,B2,B3.
  --missing-rates LIST     Missing rates. Default: 0.1,0.2,0.3,0.4,0.5.
  --dataset A,B            Dataset keys. Default: Supply_Chain_Customer_D,azure2019_D_5T.
  --imputator A,B          Imputers. Default: SAITS,DeepMVI.
  --predictor A,B          Predictors. Default: Chronos,Timer,Sundial.
  --seed 42,43             Random seeds. Default: configs/global.ini.
  --lookback 512           Lookback length L. Default: configs/global.ini.
  --k 20                   TIP sampling count. Default: configs/global.ini.
  --epsilon 1e-4           TIP finite-difference step. Default: configs/global.ini.
  --pretrain-epochs 2      B3 imputation pretraining epochs within total num_epoch. Default: configs/global.ini.
  --jobs N, -j N           Run up to N experiments concurrently. Default: 1.
  --failed-only            Skip completed experiments.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp) csv_to_array EXP_TYPES "$2"; shift 2 ;;
        --missing-rates) csv_to_array MISSING_RATES "$2"; shift 2 ;;
        --dataset) csv_to_array DATASETS "$2"; shift 2 ;;
        --imputator) csv_to_array IMPUTATORS "$2"; shift 2 ;;
        --predictor) csv_to_array PREDICTORS "$2"; shift 2 ;;
        --seed) csv_to_array SEEDS "$2"; shift 2 ;;
        --lookback) LOOKBACK_LEN="$2"; shift 2 ;;
        --k) TIP_K="$2"; shift 2 ;;
        --epsilon) TIP_EPSILON="$2"; shift 2 ;;
        --pretrain-epochs) PRETRAIN_EPOCHS="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

print_common_matrix \
    "Missing-rate robustness" \
    "${EXP_TYPES[*]}" "${DATASETS[*]}" "${IMPUTATORS[*]}" "${PREDICTORS[*]}" \
    "${SEEDS[*]}" "${MISSING_RATES[*]}" "$LOOKBACK_LEN" "$TIP_K" "$TIP_EPSILON"
echo "  B3 pretrain  : $PRETRAIN_EPOCHS epochs within total num_epoch"
echo "  dry run      : $DRY_RUN"
echo "  failed only  : $FAILED_ONLY"
echo "  jobs         : $JOBS"
echo ""

job_pool_init "$JOBS" "$DRY_RUN" "$FAILED_ONLY"

for exp_type in "${EXP_TYPES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for imputator in "${IMPUTATORS[@]}"; do
            for predictor in "${PREDICTORS[@]}"; do
                for missing_rate in "${MISSING_RATES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        job_pool_submit "$exp_type" "$dataset" "$imputator" "$predictor" \
                            "$missing_rate" "$seed" "$LOOKBACK_LEN" "$TIP_K" \
                            "$TIP_EPSILON" "$PRETRAIN_EPOCHS"
                    done
                done
            done
        done
    done
done

job_pool_wait_all
job_pool_print_summary "Missing-rate robustness summary"
