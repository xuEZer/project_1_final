#!/usr/bin/env bash

# Convenience entry point for the experiment suite described in the paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=false
FAILED_ONLY=false
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
  --jobs N, -j N       Use concurrency for all runnable experiment scripts.
  --quick              Use representative grids for ablations.
  -h, --help           Show this help.
EOF
}

QUICK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --failed-only) FAILED_ONLY=true; shift ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

COMMON_ARGS=()
[[ "$DRY_RUN" == true ]] && COMMON_ARGS+=(--dry-run)
[[ "$FAILED_ONLY" == true ]] && COMMON_ARGS+=(--failed-only)
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
[[ "$QUICK" == true ]] && COMPUTE_ARGS+=(--num-epoch 1)
bash "${SCRIPT_DIR}/run_compute_cost_probe.sh" "${COMMON_ARGS[@]}" "${COMPUTE_ARGS[@]}"

echo "[7/7] Collect results"
COLLECT_ARGS=()
if [[ "$QUICK" == true ]]; then
    COLLECT_ARGS+=(--quick --compute-num-epoch 1)
fi

if [[ "$DRY_RUN" == false ]]; then
    python3 "${SCRIPT_DIR}/collect_results.py" "${COLLECT_ARGS[@]}"
else
    echo "[DRY-RUN] python3 ${SCRIPT_DIR}/collect_results.py ${COLLECT_ARGS[*]}"
fi
