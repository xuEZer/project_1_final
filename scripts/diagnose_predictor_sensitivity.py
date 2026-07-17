#!/usr/bin/env python3
"""Diagnose whether a predictor reacts to TIP-style imputer perturbations.

The script is read-only with respect to experiments: it loads an existing B1
imputer checkpoint, uses validation batches, and never trains or saves a model.

Example:
  uv run python scripts/diagnose_predictor_sensitivity.py \
      --predictors Chronos Timer --max-batches 5 --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from configparser import ConfigParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import config_from_overrides, parse_column_index  # noqa: E402
from src.data_loader import get_tensor_data_loader, load_h5, preprocess_dataset  # noqa: E402
from src.models.imputators import imputator  # noqa: E402
from src.models.predictors import predictor  # noqa: E402
from src.rng import configure_deterministic_backend, torch_generator, torch_seed_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Supply_Chain_Customer_D")
    parser.add_argument("--imputer", default="SAITS")
    parser.add_argument(
        "--predictors", nargs="+", default=["Chronos", "Timer", "Sundial", "Toto2"]
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, help="Optionally save the summary as JSON.")
    args = parser.parse_args()
    if args.max_batches < 1 or args.k < 1 or args.baseline_repeats < 1:
        parser.error("--max-batches, --k and --baseline-repeats must be positive")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    return args


def read_best_config(dataset: str, imputer_name: str) -> tuple[Path, str, int]:
    path = ROOT / "configs" / "best_cfg" / f"best_{dataset}_{imputer_name}.ini"
    if not path.exists():
        raise FileNotFoundError(f"Best config not found: {path}")
    cfg = ConfigParser()
    cfg.read(path)
    checkpoint = Path(cfg.get("paths", "ckpt"))
    return checkpoint, cfg.get("training", "lr"), cfg.getint("loader", "batch_size")


def prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor
) -> torch.Tensor:
    observed = 1.0 - missing_mask
    return ((pred - target).square() * observed).sum() / (observed.sum() + 1e-8)


def relative_rms_change(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff_rms = (a.float() - b.float()).square().mean().sqrt()
    base_rms = torch.cat((a.float().flatten(), b.float().flatten())).square().mean().sqrt()
    return diff_rms.item(), (diff_rms / (base_rms + 1e-12)).item()


@torch.inference_mode()
def diagnose_predictor(
    *, config, imputer_model, predictor_name: str, val_loader, args
) -> dict[str, float | int | str]:
    device = config.get("training", "device")
    pred_model = predictor(predictor_name, config)
    pred_model.eval()
    pred_model.to(device)
    imputer_model.eval()

    params = list(imputer_model.parameters())
    bases = [p.detach().clone() for p in params]
    direction_rng = torch_generator(
        args.seed, "sensitivity_directions", predictor_name, device=device
    )
    baseline_losses: list[float] = []
    baseline_noise_variances: list[float] = []
    baseline_pred_changes: list[float] = []
    loss_differences: list[float] = []
    abs_directional_derivatives: list[float] = []
    prediction_changes: list[float] = []
    relative_prediction_changes: list[float] = []

    try:
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                tensor.to(device) for tensor in batch
            )
            baseline_imputed = imputer_model(
                X_NA, X_NA_mask, X_N, X_A_mask
            )["imp_data"]
            repeated_predictions = [
                pred_model(baseline_imputed)["pred_data"]
                for _ in range(args.baseline_repeats)
            ]
            repeated_losses = [
                prediction_loss(pred, Y_N, Y_N_mask).item()
                for pred in repeated_predictions
            ]
            baseline_losses.extend(repeated_losses)
            if len(repeated_losses) > 1:
                baseline_noise_variances.append(
                    torch.tensor(repeated_losses, dtype=torch.float64)
                    .var(unbiased=True)
                    .item()
                )
            anchor = repeated_predictions[0]
            baseline_pred_changes.extend(
                relative_rms_change(anchor, pred)[0]
                for pred in repeated_predictions[1:]
            )

            directions: list[list[torch.Tensor]] = []
            imputed_pairs: list[torch.Tensor] = []
            for _ in range(args.k):
                current = [
                    torch.empty_like(p).normal_(generator=direction_rng) for p in params
                ]
                directions.append(current)
                for p, base, direction in zip(params, bases, current, strict=True):
                    p.copy_(base + args.epsilon * direction)
                imputed_pairs.append(
                    imputer_model(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )
                for p, base, direction in zip(params, bases, current, strict=True):
                    p.copy_(base - args.epsilon * direction)
                imputed_pairs.append(
                    imputer_model(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )
                for p, base in zip(params, bases, strict=True):
                    p.copy_(base)

            batch_size = X_NA.shape[0]
            paired_predictions = pred_model(torch.cat(imputed_pairs))["pred_data"]
            for i in range(args.k):
                plus = paired_predictions[2 * i * batch_size : (2 * i + 1) * batch_size]
                minus = paired_predictions[
                    (2 * i + 1) * batch_size : (2 * i + 2) * batch_size
                ]
                plus_loss = prediction_loss(plus, Y_N, Y_N_mask).item()
                minus_loss = prediction_loss(minus, Y_N, Y_N_mask).item()
                loss_diff = plus_loss - minus_loss
                absolute, relative = relative_rms_change(plus, minus)
                loss_differences.append(loss_diff)
                abs_directional_derivatives.append(abs(loss_diff) / (2 * args.epsilon))
                prediction_changes.append(absolute)
                relative_prediction_changes.append(relative)
    finally:
        for p, base in zip(params, bases, strict=True):
            p.copy_(base)
        del pred_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    losses = torch.tensor(baseline_losses, dtype=torch.float64)
    differences = torch.tensor(loss_differences, dtype=torch.float64)
    noise_std = (
        (sum(baseline_noise_variances) / len(baseline_noise_variances)) ** 0.5
        if baseline_noise_variances else 0.0
    )
    median_abs_diff = differences.abs().median().item()
    snr = median_abs_diff / ((2.0**0.5) * noise_std) if noise_std > 0 else float("inf")
    return {
        "predictor": predictor_name,
        "batches": min(args.max_batches, len(val_loader)),
        "directions": len(loss_differences),
        "baseline_loss_mean": losses.mean().item(),
        "baseline_loss_std": noise_std,
        "baseline_prediction_noise_rms_mean": (
            sum(baseline_pred_changes) / len(baseline_pred_changes)
            if baseline_pred_changes else 0.0
        ),
        "mean_abs_directional_derivative": sum(abs_directional_derivatives)
        / len(abs_directional_derivatives),
        "median_abs_loss_difference": median_abs_diff,
        "paired_difference_snr": snr,
        "prediction_change_rms_mean": sum(prediction_changes) / len(prediction_changes),
        "relative_prediction_change_mean": sum(relative_prediction_changes)
        / len(relative_prediction_changes),
    }


def print_result(result: dict[str, float | int | str]) -> None:
    print(f"\n[{result['predictor']}] batches={result['batches']} "
          f"directions={result['directions']}")
    for key in (
        "baseline_loss_mean", "baseline_loss_std",
        "baseline_prediction_noise_rms_mean", "mean_abs_directional_derivative",
        "median_abs_loss_difference", "paired_difference_snr",
        "prediction_change_rms_mean", "relative_prediction_change_mean",
    ):
        print(f"  {key:38s} {float(result[key]):.6g}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("sensitivity")
    configure_deterministic_backend()

    default_ckpt, best_lr, best_batch_size = read_best_config(args.dataset, args.imputer)
    checkpoint = (args.checkpoint or default_ckpt).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    batch_size = args.batch_size or best_batch_size

    def overrides_for(predictor_name: str) -> list[str]:
        overrides = [
            "--data.dataset", args.dataset,
            "--imputer.model", args.imputer,
            "--predictor.model", predictor_name,
            "--training.lr", best_lr,
            "--loader.batch_size", str(batch_size),
            "--experiment.seed", str(args.seed),
        ]
        if args.device:
            overrides += ["--training.device", args.device]
        return overrides

    config = config_from_overrides(overrides_for(args.predictors[0]))

    dataset_folder = preprocess_dataset(
        source=config.get("dataset", "source"),
        repo_name=config.get("dataset", "repo_name"),
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.get("dataset", "item_id"),
        column_index=parse_column_index(config.get("dataset", "column_index")),
        missing_rate=config.getfloat("data", "missing_rate"),
        artificial_missing_rate=config.getfloat("data", "artificial_missing_rate"),
        block_size=config.getint("data", "block_size"),
        lookback_len=config.getint("data", "lookback"),
        pred_len=config.getint("data", "pred_len"),
        val_ratio=config.getfloat("data", "val_ratio"),
        test_ratio=config.getfloat("data", "test_ratio"),
        random_seed=args.seed,
        logger=logger,
        save_path=Path(config.get("paths", "cache_dir")) / "data",
    )
    dataset = load_h5(dataset_folder / "dataset.h5")
    config.set("data", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("data", "lookback", str(dataset["train"]["X_N"].shape[1]))
    val_loader = get_tensor_data_loader(dataset, batch_size=batch_size)[0].get("val")
    if val_loader is None:
        raise RuntimeError("The selected dataset has no training-format validation split")

    device = config.get("training", "device")
    with torch_seed_scope(args.seed, "sensitivity_imputer", args.imputer):
        imputer_model = imputator(args.imputer, config).to(device)
    state = torch.load(checkpoint, map_location=device)
    imputer_model.load_state_dict(state)
    imputer_model.eval()

    print(f"Checkpoint: {checkpoint}")
    print(f"Validation batch size: {batch_size}; k={args.k}; epsilon={args.epsilon:g}")
    results = []
    for predictor_name in args.predictors:
        predictor_config = config_from_overrides(overrides_for(predictor_name))
        predictor_config.set("data", "d_feature", config.get("data", "d_feature"))
        predictor_config.set("data", "lookback", config.get("data", "lookback"))
        result = diagnose_predictor(
            config=predictor_config,
            imputer_model=imputer_model,
            predictor_name=predictor_name,
            val_loader=val_loader,
            args=args,
        )
        results.append(result)
        print_result(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
        print(f"\nSaved JSON: {args.json}")


if __name__ == "__main__":
    main()
