# PILOT

Time-series imputation with a frozen forecaster. B1 trains the imputer on reconstruction; B2 applies PILOT after a shared warm-up.

## Setup

```bash
uv sync
cd src/models/predictors/Sundial && uv sync && cd -
```

The root environment includes the official TimesFM 3 PyTorch runtime. For users in mainland China, set `HF_ENDPOINT=https://hf-mirror.com`.

## Experiments

```bash
python run_expman.py --staged --config configs/experiment.yaml
python run_expman.py --staged --config configs/ablation.yaml
python run_expman.py --staged --config configs/sensitivity.yaml
```

