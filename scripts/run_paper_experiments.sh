#!/usr/bin/env bash

# Convenience entry point for the experiment suite described in the paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=false
FAILED_ONLY=false
DEVICE=""
MAX_EPOCH=""
PATIENCE=""
BATCH_SIZE=""
JOBS=""

usage() {
    cat <<'EOF'
Usage: scripts/run_paper_experiments.sh [options]

Runs:
  1. Main B1/B2 experiment.
  2. Projection ablation B4/B5/B6 (B1/B2 are produced by step 1).
  3. k sweep.
  4. epsilon sweep.
  5. Missing-rate robustness with B1/B2/B3.
  6. Compute-cost probe.
  7. Result collection into results/experiment_results.csv.

Options:
  --dry-run            Print commands without running training.
  --failed-only        Skip completed logs.
  --device cuda|cpu    Override Torch device.
  --max-epoch N        Override max training epochs.
  --patience N         Override early-stopping patience.
  --batch-size N       Override effective batch size.
  --jobs N, -j N       Use concurrency for all runnable experiment scripts.
  --quick              Use representative grids for ablations and 2 max epochs.
  -h, --help           Show this help.
EOF
}

QUICK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --max-epoch) MAX_EPOCH="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$QUICK" == true ]]; then
    [[ -z "$MAX_EPOCH" ]] && MAX_EPOCH=2
    [[ -z "$PATIENCE" ]] && PATIENCE=1
fi

COMMON_ARGS=()
[[ "$DRY_RUN" == true ]] && COMMON_ARGS+=(--dry-run)
[[ "$FAILED_ONLY" == true ]] && COMMON_ARGS+=(--failed-only)
[[ -n "$DEVICE" ]] && COMMON_ARGS+=(--device "$DEVICE")
[[ -n "$MAX_EPOCH" ]] && COMMON_ARGS+=(--max-epoch "$MAX_EPOCH")
[[ -n "$PATIENCE" ]] && COMMON_ARGS+=(--patience "$PATIENCE")
[[ -n "$BATCH_SIZE" ]] && COMMON_ARGS+=(--batch-size "$BATCH_SIZE")
RUNNABLE_ARGS=("${COMMON_ARGS[@]}")
[[ -n "$JOBS" ]] && RUNNABLE_ARGS+=(--jobs "$JOBS")

echo "[1/7] Main B1/B2 experiment"
bash "${SCRIPT_DIR}/run_experiments.sh" "${RUNNABLE_ARGS[@]}"

echo "[2/7] Projection ablation B4/B5/B6"
ABLATION_ARGS=(--exp B4,B5,B6)
[[ "$QUICK" == true ]] && ABLATION_ARGS+=(--representative)
bash "${SCRIPT_DIR}/run_projection_ablation.sh" "${RUNNABLE_ARGS[@]}" "${ABLATION_ARGS[@]}"

echo "[3/7] K sweep"
K_SWEEP_ARGS=()
[[ "$QUICK" == true ]] && K_SWEEP_ARGS+=(--k-values 5,20)
bash "${SCRIPT_DIR}/run_k_sweep.sh" "${RUNNABLE_ARGS[@]}" "${K_SWEEP_ARGS[@]}"

echo "[4/7] Epsilon sweep"
EPS_SWEEP_ARGS=()
[[ "$QUICK" == true ]] && EPS_SWEEP_ARGS+=(--epsilon-values 1e-4,1e-3)
bash "${SCRIPT_DIR}/run_epsilon_sweep.sh" "${RUNNABLE_ARGS[@]}" "${EPS_SWEEP_ARGS[@]}"

echo "[5/7] Missing-rate robustness"
ROBUST_ARGS=()
[[ "$QUICK" == true ]] && ROBUST_ARGS+=(--missing-rates 0.2,0.4 --exp B1,B2,B3)
bash "${SCRIPT_DIR}/run_missing_rate_robustness.sh" "${RUNNABLE_ARGS[@]}" "${ROBUST_ARGS[@]}"

echo "[6/7] Compute-cost probe"
COMPUTE_ARGS=()
[[ "$QUICK" == true ]] && COMPUTE_ARGS+=(--max-epoch 1)
bash "${SCRIPT_DIR}/run_compute_cost_probe.sh" "${COMMON_ARGS[@]}" "${COMPUTE_ARGS[@]}"

echo "[7/7] Collect results"
COLLECT_ARGS=()
if [[ "$QUICK" == true ]]; then
    COLLECT_ARGS+=(--quick --compute-max-epoch 1)
elif [[ -n "$MAX_EPOCH" ]]; then
    COLLECT_ARGS+=(--compute-max-epoch "$MAX_EPOCH")
fi

if [[ "$DRY_RUN" == false ]]; then
    python3 "${SCRIPT_DIR}/collect_results.py" "${COLLECT_ARGS[@]}"
else
    echo "[DRY-RUN] python3 ${SCRIPT_DIR}/collect_results.py ${COLLECT_ARGS[*]}"
fi
