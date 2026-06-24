# FOTSI

Time-series forecasting with imputation. Impute-then-predict (B1) and TIP joint training (B2).

## Setup

```bash
uv sync
cd src/models/predictors/Sundial && uv sync && cd -
cd src/models/predictors/TimesFM25 && uv sync && cd -
```

For users in mainland China:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Run

```bash
./scripts/run_experiments.sh                  # B1 + B2, all 7 datasets
./scripts/run_experiments.sh --failed-only    # skip completed
./scripts/run_experiments.sh --dataset CPHL_30T --exp B2
./scripts/run_k_sweep.sh                      # TIP k ablation
./scripts/run_epsilon_sweep.sh                # TIP epsilon ablation
./scripts/run_missing_rate_robustness.sh      # B3 missing-rate robustness
```
