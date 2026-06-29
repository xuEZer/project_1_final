# Experiment scripts

These scripts are generated from the experiment design in `C:\Users\29895\Desktop\research_doc\main.tex`.

## Main entry points

- `run_experiments.sh`: main B1 vs B2 experiment on all 7 datasets, 4 imputers, and 2 TSFMs.
- `run_projection_ablation.sh`: B1/B2/B4/B5/B6 projection-strategy ablation.
- `run_k_sweep.sh`: TIP sampling-count ablation, `k in {5, 10, 20, 50, 100}`.
- `run_epsilon_sweep.sh`: TIP perturbation-step ablation, `epsilon in {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`.
- `run_missing_rate_robustness.sh`: missing-rate robustness with B1/B2/B3 on Supply Chain-Customer and Azure2019-D.
- `run_compute_cost_probe.sh`: short wall-clock probe for the compute-cost table.
- `run_paper_experiments.sh`: convenience wrapper that runs the full suite.
- `collect_results.py`: parses `logs/*.log` and writes CSV/statistical summaries into `results/`.

## Examples

```bash
bash scripts/run_experiments.sh --dry-run
bash scripts/run_experiments.sh --dataset CPHL_30T --exp B2
bash scripts/run_projection_ablation.sh --representative --dry-run
bash scripts/run_paper_experiments.sh --quick --dry-run
python3 scripts/collect_results.py
```

## Paper defaults

- Missing rate for main and ablation experiments: `0.2`
- Lookback length: `128`
- Block missing size: `20`
- Batch size: `128`
- Optimizer settings are taken from `configs/global.ini`: AdamW, learning rate `1e-4`
- TIP defaults: `k=20`, `epsilon=1e-4`
- B3 pretraining: `2` epochs
