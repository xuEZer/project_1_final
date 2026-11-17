# PILOT

PILOT trains an imputer around a frozen time-series forecaster through forward queries. It measures forecast-loss derivatives in a subspace containing the reconstruction update and recent applied directions, estimates a finite-population confidence ellipsoid, and applies a robust release test or a minimum-change cone projection. Every nonzero projected correction has the reconstruction update's norm.

## Experiments

The experiment runtime is included in `expman/` and installed with this project by `uv sync`. The `expman` command and `python -m expman` use this local implementation. The bundled version is `0.1.1+fixed1g` and includes its dashboard assets. Running batches requires Linux, an NVIDIA GPU, and NVIDIA MPS (`nvidia-cuda-mps-control`).

`configs/main.yaml` defines all experiments through the single pipeline:

```bash
uv sync --frozen
expman run src.pipeline:build_pipeline configs/main.yaml
expman status
expman web
expman resume <batch-id>
```

The batch contains 223 unique configurations. The main experiment crosses five datasets (EWELD Load, Coastal T–S, Port Activity, Amazon Supply Chain, Current Velocity), five imputers (SAITS, DeepMVI, ModernTCN, TimesNet, TSLANet), and four forecasters (Toto2, Chronos-2, Sundial, TimesFM3), each trained with reconstruction alone and PILOT: 200 runs.

Component ablations use Current Velocity / ModernTCN / Chronos-2. Their reconstruction and full-PILOT reference runs are shared with the main experiment.

| Variant | `training.method` | Update |
| --- | --- | --- |
| B1 | `raw` | Reconstruction optimizer displacement |
| B2 | `pilot` | Trajectory subspace, robust release, cone projection |
| B3 | `random` | Random companion directions in the candidate-containing subspace |
| B4 | `point` | Point-estimate halfspace projection |
| B5 | `mean_gate` | Mean-sign release with robust projection on rejection |
| B6 | `forecast_direction` | Measured forecast-gradient direction on rejection |

The sensitivity study uses the same data, models, and seed. Its 19 configurations each change exactly one parameter from B2; the default B2 run is shared.

| Parameter | B2 default | Sensitivity values |
| --- | --- | --- |
| Subspace rank | 3 | 2, 4, 5, 6 |
| Finite-difference step | 0.001 | 0.0001, 0.0002, 0.0005, 0.002, 0.005, 0.01 |
| Miscoverage | 0.3 | 0.05, 0.1, 0.2, 0.4, 0.5 |
| Probe batches | 4 | 2, 3, 5, 6 |

## Protocol

The selected complete univariate series receive length-8 missing blocks at rate 0.2, followed by independent artificial masking at rate 0.2 on observed entries. Windows use 256 context steps and an 8-step horizon, with a chronological 7:1:2 split by window count. Standardization uses observed training values only.

Training uses seed 42, AdamW at learning rate 0.001, batch size 128, and at most 50 epochs. Validation forecast MSE controls stopping with patience 5. The final continuation state is evaluated on the complete train, validation, and test splits. The forecaster remains frozen. PILOT starts from the same random imputer initialization as reconstruction-only training, without a separate warmup phase.

`training.batch_size` controls training, evaluation, and probing. Four probe batches request 512 windows sampled uniformly without replacement from the complement of the current optimizer batch, capped by the residual population. Insufficient subspace rank or fewer than two probe windows releases the reconstruction update. Common random numbers couple the positive and negative finite-difference queries. Sundial uses a fixed sampling stream per window, so its forecast is invariant to query order and batch composition.

All runs use GPU 0; expman schedules concurrent workers according to available memory. Model revisions and numerical precision are declared in each predictor's configuration.

## Configuration and artifacts

```text
configs/
├── main.yaml       # Main experiment, component ablations, one-factor sensitivity
├── base.yaml       # Shared data, training, and PILOT protocol
├── dataset/        # Dataset source and selected series
└── models/
    ├── imputer/    # Imputer names and architecture parameters
    └── predictor/  # Forecaster names, revisions, precision, and query parameters
```

The experiment YAML references datasets and models through `defaults`. Its root `combines` list specifies independent complete configurations using expman's native expansion. Training and evaluation metrics are recorded through `ctx.log_metrics()` in each batch's `metrics.sqlite3`; model weights and source/configuration metadata belong to the run's artifacts. Expman owns checkpoint state and batch resumption.

Use `scripts/plot_results.py` to create figures from recorded results. Experiment artifacts belong to `runs/`, data and downloaded weights to `data/`, temporary validation to `tmp/`, and final figures to `paper/figures/`. The manuscript in `paper/` is an independent repository.

Run `ruff check .` and `ruff format --check .` before committing.
