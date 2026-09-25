# PILOT

**P**robe-based **I**mputation via **L**ocal Subspace **O**ptimization for **T**ime-Series Forecasting.

PILOT trains an imputer for a frozen, inference-only time-series foundation model
(TSFM) without backpropagating through the forecaster. At each optimizer step it
builds a local subspace from the optimizer's candidate descent direction and
recently applied directions, measures forecast-gradient coordinates in that
subspace with `2r` symmetric finite-difference queries over the current batch
plus a without-replacement probe sample, forms a **finite-population estimator**
of the full-training gradient mean, and applies a **safe-release** rule: the raw
update is released only when it is robustly non-harming for a confidence
ellipsoid of the unobserved full-training mean; otherwise a minimum-change
robust correction is applied. A shared reconstruction warm-up trains the
imputer alone first; the B1/B2 branches then continue from the *same* captured
checkpoint (identical weights, optimizer, and RNG state), so the two arms
differ only in whether the forecast mechanism intervenes.

This repository contains the code for the controlled mechanism validation
(RQ0), the main imputer–forecaster–dataset matrix (RQ2–RQ3), the component
ablation, and the parameter-sensitivity scans.

---

## Repository layout

Only the following are needed to reproduce the paper's experiments:

```
run_expman.py               Entry point (expman batch runner; --staged protocol)
configs/
  experiment.yaml           Main matrix: 2 stages x 5 datasets x 5 imputers x 4 predictors (200 runs)
  ablation.yaml             Component ablation: 6 branches on one representative cell (6 runs)
  ablation_base.yaml        Shared protocol for ablation/sensitivity (cell + tip + staged protocol)
  sensitivity.yaml          OFAT sensitivity, 19 one-factor B2 arms + B1 reference (20 runs)
  sensitivity_refined.yaml  Additional one-factor points (epsilon, probe count)
  sensitivity_reference.yaml Alignment point for the refined scan
  dataset/  imputer/  predictor/   Named defaults resolved by expman (source, item, model hparams)
src/
  staged_pipeline.py        Warmup / ContinueTraining / Evaluate stages (shared-checkpoint protocol)
  pipeline.py               expman pipeline (data prep, config, run tags)
  experiments.py            B1--B5, B7 branch definitions and training drivers
  models/jointModel.py      The PILOT mechanism: subspace, finite-difference probing,
                            finite-population ellipsoid, safe-release gate, robust correction
  data_loader.py            Windowing, shuffled 7:1:2 split, block-missing generation
  evaluator.py              Test/validation metrics (forecast MSE on observed future positions)
scripts/
  run_rq0_controlled.py     Fully controlled mechanism validation (analytic objectives, 6-panel figure)
  run_pilot_controlled.py   Paired single-step controlled experiment (population seeds, probe draws)
figures/plot_*.py           Optional analysis/plotting utilities used for the paper's figures
```

`runs/`, `results/`, `logs/`, `cache/` and `tmp/` are runtime artifacts and are
git-ignored. `scripts/check_scoping.py` is a development utility, not part of
the reproduction path.

---

## Setup

Requirements: Linux, Python 3.12 (managed by `uv`), one CUDA GPU (the paper's
runs used a single 24 GB consumer card), and Hugging Face access for the
datasets (`Real-TSF/TIME`) and the frozen forecasters (TimesFM 3, Chronos-2;
Toto2 ships with the `toto-models` package; Sundial is a local module under
`src/models/predictors/Sundial`).

```bash
uv sync                      # root environment (includes the TimesFM 3 PyTorch runtime)
```

For users in mainland China, point Hugging Face at a mirror **before** the
first run (the runner also sets this by default):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

`run_expman.py` pins BLAS/torch thread pools and idle-pool waiting policy for
its worker processes automatically; no manual tuning is needed.

---

## The staged protocol

All three expman matrices below must be launched with `--staged`. The staged
pipeline (`src/staged_pipeline.py`) is:

1. **PrepareData** — window the series (lookback 256, horizon 8), apply the
   shuffled 7:1:2 split and the missingness protocol (contiguous blocks of 8
   at rate 0.2, plus a 20% pointwise artificial mask on observed positions;
   `missing_seed` fixes the realization across seeds).
2. **Warmup** — reconstruction-only training with AdamW (lr 1e-3, batch 128).
   Stops after `patience: 5` epochs without a new minimum validation forecast
   loss; captures the best checkpoint **including optimizer and all RNG
   states**.
3. **ContinueTraining** — each branch (B1 raw / B2 PILOT / ...) restores the
   identical captured state and trains exactly `continuation_epochs: 1` epoch.
   B2 engages the forecast mechanism (subspace rank 3, ε=1e-3 FD probes over
   the current batch + 4 probe batches, `safe_release` gate with Hotelling
   ellipsoid radius, `cone_delta 0.3`, `preserve_norm` so corrected steps keep
   the optimizer's displacement norm).
4. **Evaluate** — test-set imputation/forecast metrics, forecast MSE computed
   on observed future positions. The final-epoch imputer is delivered
   directly (no checkpoint selection).

Branches (see `src/experiments.py: variants`): **B1** reconstruction only;
**B2** full PILOT; **B3** random subspace basis; **B4** point projection
(no uncertainty margin); **B5** mean gate (no safe-release margin);
**B7** direct application of the estimated forecast-gradient direction
(norm-preserving). The paired design means every B1/B2 difference at the
endpoint is attributable to the mechanism alone.

---

## Running the experiments

Every batch is resumable: re-running the same command, or
`python run_expman.py --resume --output-dir runs/<name>`, skips completed runs.
Progress lines print to stdout; per-run metrics accumulate in
`runs/<name>/metrics.sqlite3` (table `metrics`, JSON `metric_path` such as
`["test", "pred_mse"]`, stage `[1]` = continuation training, `[2]` = evaluation;
run metadata in `runs/<name>/experiments/<id>/config.pkl`).

### 1. Main matrix — RQ2/RQ3 (200 runs)

```bash
python run_expman.py --staged --config configs/experiment.yaml --output-dir runs/main
```

2 stages × 5 datasets (`EWELD_Load_15T`, `Coastal_T_S_15T`, `Port_Activity_D`,
`Supply_Chain_Customer_D_Amazon`, `current_velocity_10T`) × 5 imputers
(SAITS, DeepMVI, ModernTCN, TimesNet, TSLANet) × 4 frozen forecasters (Toto2,
Chronos, Sundial, TimesFM3), `seed = missing_seed = 42`. This is the long
pole: heavy forecaster cells dominate; on a single 24 GB GPU plan for the
batch to run for many hours — use `--resume` across sessions.

### 2. Component ablation (6 runs)

```bash
python run_expman.py --staged --config configs/ablation.yaml --output-dir runs/ablation
```

B1/B2/B3/B4/B5/B7 on the representative cell
`current_velocity_10T / ModernTCN / Chronos`, seed 42, sharing the staged
warm-up — each branch is one continuation epoch from the same checkpoint.

### 3. Parameter sensitivity (~27 runs)

```bash
python run_expman.py --staged --config configs/sensitivity.yaml           --output-dir runs/sensitivity
python run_expman.py --staged --config configs/sensitivity_refined.yaml   --output-dir runs/sensitivity_refined
python run_expman.py --staged --config configs/sensitivity_reference.yaml --output-dir runs/sensitivity_reference
```

One-factor-at-a-time arms around the base point (subspace rank, ε, gate δ,
cone δ, probe-batch count), all on the representative cell with the B1
reference in-batch for paired comparison.

### 4. Controlled mechanism validation — RQ0/RQ1 (no GPU needed)

```bash
python scripts/run_rq0_controlled.py --output-dir results/rq0
python scripts/run_pilot_controlled.py --output-dir runs/controlled_pilot
```

Fully analytic, paired-seed single-step experiments (the mechanism's six-panel
figure, conflict boundary, FD scaling, finite-population coverage), CPU-only
and fast. `run_pilot_controlled.py` additionally exposes
`--population-seeds` and `--probe-draws` for the coverage diagnostics.

---

## Reading the results

- **Per-run endpoints**: `runs/<name>/metrics.sqlite3`, metric path
  `["test", "pred_mse"]` / `["test", "imp_mse"]` at stage `[2]` (latest
  attempt). Join with `experiments/<id>/config.pkl` for
  (dataset, imputer, predictor, stage, seed).
- **Process traces**: stage `[1]` rows include per-epoch
  `["loss", "train/val_imp_loss"]`, `["loss", "train/val_pred_loss"]` and
  `["tip", <branch>, ...]` diagnostics (projection rate, robust margin,
  correction ratio, finite-population coverage statistics).
- **Aggregates for the paper**: `results/` holds the CSVs behind the paper's
  tables (paired non-increase rates, endpoint losses); `figures/plot_*.py`
  regenerate the corresponding figures from a metrics snapshot.

## Reproducibility notes

- Fixed horizons and no checkpoint selection: endpoints reflect the delivered
  final-epoch model.
- `seed` controls initialization, batch order, dropout and all probing
  randomness; `missing_seed` (equal to `seed` here) fixes the missingness
  realization. B1/B2 pairs share both.
- Forecaster weights are pinned by the locked dependencies (`uv.lock`,
  expman revision) and downloaded once into the HF cache.
