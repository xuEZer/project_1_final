# Experiment scripts

These scripts are generated from the experiment design in `C:\Users\29895\Desktop\research_doc\main.tex`.

## Main entry points

- `run_experiments.sh`: main B1 vs B2 experiment on all 7 datasets, 4 imputers, and 2 TSFMs.
- `run_core_validation.sh`: first-layer B1/B2 validation on a small affordable
  matrix before full experiments.
- `run_projection_ablation.sh`: B1/B2/B4/B5/B6 projection-strategy ablation.
- `run_k_sweep.sh`: TIP sampling-count ablation, `k in {5, 10, 20, 50, 100}`.
- `run_epsilon_sweep.sh`: TIP perturbation-step ablation, `epsilon in {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`.
- `run_missing_rate_robustness.sh`: missing-rate robustness with B1/B2/B3 on Supply Chain-Customer and Azure2019-D.
- `run_compute_cost_probe.sh`: short wall-clock probe for the compute-cost table.
- `profile_runtime.py`: profile startup, data, model init, imputer training,
  predictor inference, and TIP zero-order estimation for one config.
- `run_paper_experiments.sh`: convenience wrapper that runs the full suite.
- `auto_run_experiments.py`: budget-aware local runner that probes runtime,
  monitors logs, retries smaller batches on OOM, downgrades pending grids when
  the selected profile no longer fits, and collects results.
- `collect_results.py`: parses `logs/*.log` and writes CSV/statistical summaries into `results/`.

## Examples

```bash
bash scripts/run_experiments.sh --dry-run
bash scripts/run_core_validation.sh --dry-run
bash scripts/run_core_validation.sh --failed-only --jobs 1
bash scripts/run_core_validation.sh --with-sensitivity --dry-run
bash scripts/run_core_validation.sh --sensitivity-only --epsilon-k 10
bash scripts/run_experiments.sh --dataset CPHL_30T --exp B2
bash scripts/run_projection_ablation.sh --representative --dry-run
bash scripts/run_paper_experiments.sh --quick --dry-run
python3 scripts/auto_run_experiments.py --budget-min 90 --profile auto
python3 scripts/auto_run_experiments.py --budget-min 20 --profile smoke --device cuda
python3 scripts/collect_results.py
./src/models/predictors/Sundial/.venv/bin/python scripts/profile_runtime.py --batches 3 --tip-k 5 --dataset.dataset Supply_Chain_Customer_D --imputator.model_type SAITS --predictor.model_type Sundial
```

## Runtime profiling

Use `profile_runtime.py` when an experiment feels slow and you need to locate
the bottleneck before changing code or hyperparameters.  It runs only a few
batches and reports per-stage timings plus rough B1/TIP epoch estimates.

Examples:

```bash
./src/models/predictors/Sundial/.venv/bin/python scripts/profile_runtime.py --batches 3 --tip-k 5
./src/models/predictors/TimesFM25/.venv/bin/python scripts/profile_runtime.py --batches 2 --tip-k 5 --predictor.model_type TimesFM25
./src/models/predictors/Sundial/.venv/bin/python scripts/profile_runtime.py --skip-tip --include-auto-batch
```

## Adaptive local runner

Use `auto_run_experiments.py` when the machine budget is uncertain.  The
default `--profile auto` first runs a tiny smoke grid, estimates runtime from
the observed jobs, then chooses the largest profile that should fit the
remaining budget:

- `smoke`: B1/B2 on Supply_Chain_Customer_D + SAITS + Sundial, L=128.
- `quick`: 3 datasets + SAITS + Sundial, short early-stopped runs.
- `balanced`: 7 datasets + BRITS/SAITS + Sundial, moderate L=256 runs.
- `paper`: full B1/B2 paper matrix.

Useful options:

```bash
python3 scripts/auto_run_experiments.py --budget-min 60 --profile auto --device cuda
python3 scripts/auto_run_experiments.py --budget-min 15 --profile smoke --allow-cpu-fallback
python3 scripts/auto_run_experiments.py --profile quick --dataset CPHL_30T,azure2019_D_5T
```

## First-Layer Validation

Run this before full paper experiments:

```bash
bash scripts/run_core_validation.sh --failed-only --jobs 1
```

The default grid is intentionally small: `B1,B2` on three representative
datasets, `SAITS,DeepMVI`, `Sundial`, `lookback=128`, `k=5`, `max_epoch=5`,
and `patience=2`.  This is meant to answer the first question: whether TIP
shows a consistent signal over B1 under an affordable budget.

The same script can also run a small hyperparameter sensitivity probe:

```bash
bash scripts/run_core_validation.sh --with-sensitivity --failed-only --jobs 1
```

This adds B2-only sweeps on the same small matrix:

- `k in {5,10,20}` with `epsilon=1e-4`.
- `epsilon in {1e-5,1e-4,1e-3}` with fixed `k`, controlled by `--epsilon-k`.

After inspecting the k sweep, you can rerun only the epsilon probe with the
chosen k:

```bash
bash scripts/run_core_validation.sh --sensitivity-only --epsilon-k 10 --failed-only
```

## Paper defaults

- Missing rate for main and ablation experiments: `0.2`
- Lookback length: `128`
- Block missing size: `20`
- Batch size: `128`
- Optimizer settings are taken from `configs/global.ini`: AdamW, learning rate `1e-4`
- TIP defaults: `k=20`, `epsilon=1e-4`
- B3 pretraining: `2` epochs
