#!/usr/bin/env bash

# Short wall-clock probe for the compute-cost table.
# This measures short training runs under identical data/model settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common_experiment.sh
source "${SCRIPT_DIR}/common_experiment.sh"

DATASET="Supply_Chain_Customer_D"
IMPUTATOR="SAITS"
PREDICTOR="TimesFM25"
MISSING_RATE="$DEFAULT_MAIN_MISSING_RATE"
LOOKBACK_LEN="$DEFAULT_LOOKBACK_LEN"
SEED=42
TIP_EPSILON="$DEFAULT_TIP_EPSILON"
PRETRAIN_EPOCHS="$DEFAULT_PRETRAIN_EPOCHS"
MAX_EPOCH=1
PATIENCE=1
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
DEVICE="$DEFAULT_DEVICE"
DRY_RUN=false
FAILED_ONLY=false
OUT_CSV="${RESULT_DIR}/compute_cost_probe.csv"

usage() {
    cat <<'EOF'
Usage: scripts/run_compute_cost_probe.sh [options]

Options:
  --dataset KEY          Dataset config key. Default: Supply_Chain_Customer_D.
  --imputator NAME       Imputer. Default: SAITS.
  --predictor NAME       Predictor. Default: TimesFM25.
  --missing-rate 0.2     Missing rate. Default: 0.2.
  --lookback 128         Lookback length L. Default: 128.
  --seed 42              Random seed. Default: 42.
  --epsilon 1e-4         TIP finite-difference step. Default: 1e-4.
  --max-epoch N          Short run epochs. Default: 1.
  --patience N           Early-stopping patience. Default: 1.
  --batch-size N         Effective batch size. Default: 128.
  --device cuda|cpu      Torch device. Default: cuda.
  --failed-only          Skip completed probe logs.
  --dry-run              Print commands without running them.
  -h, --help             Show this help.

Probe rows:
  B1
  B2 k=20
  B2 k=10
  B2 k=5
  B6 k=20
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --imputator) IMPUTATOR="$2"; shift 2 ;;
        --predictor) PREDICTOR="$2"; shift 2 ;;
        --missing-rate) MISSING_RATE="$2"; shift 2 ;;
        --lookback) LOOKBACK_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --epsilon) TIP_EPSILON="$2"; shift 2 ;;
        --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_DIR"

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$RESULT_DIR"
    echo "strategy,k,elapsed_seconds,exit_code" > "$OUT_CSV"
fi
export EXP_NAME_SUFFIX="costprobe_e${MAX_EPOCH}"

run_timed() {
    local exp_type="$1"
    local k="$2"
    local start elapsed code
    start="$(date +%s)"
    set +e
    run_one_experiment "$exp_type" "$DATASET" "$IMPUTATOR" "$PREDICTOR" \
        "$MISSING_RATE" "$SEED" "$LOOKBACK_LEN" "$k" "$TIP_EPSILON" \
        "$PRETRAIN_EPOCHS" "$DRY_RUN" "$FAILED_ONLY" "$MAX_EPOCH" \
        "$PATIENCE" "$BATCH_SIZE" "$DEVICE"
    code=$?
    set -e
    elapsed=$(( $(date +%s) - start ))
    if [[ "$DRY_RUN" == false ]]; then
        echo "${exp_type},${k},${elapsed},${code}" >> "$OUT_CSV"
    fi
    return "$code"
}

echo "=============================================="
echo "  Compute-cost probe"
echo "=============================================="
echo "  config       : $DATASET + $IMPUTATOR + $PREDICTOR"
echo "  max epoch    : $MAX_EPOCH"
echo "  output       : $OUT_CSV"
echo "=============================================="

run_timed B1 0
run_timed B2 20
run_timed B2 10
run_timed B2 5
run_timed B6 20

if [[ "$DRY_RUN" == false ]]; then
    echo "Wrote $OUT_CSV"
else
    echo "Dry run complete; no CSV written."
fi
