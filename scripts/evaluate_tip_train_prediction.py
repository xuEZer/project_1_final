#!/usr/bin/env python3
"""Evaluate training-set prediction MSE for every TIP grid checkpoint.

The original tuning result is never modified. Results are checkpointed after
each model evaluation, so an interrupted run can be resumed safely.

Usage:
  uv run python scripts/evaluate_tip_train_prediction.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import config_from_overrides, resolve_experiment_name  # noqa: E402
from src.data_loader import load_h5, preprocess_dataset  # noqa: E402
from src.models.imputators import imputator  # noqa: E402
from src.models.predictors import predictor  # noqa: E402
from src.rng import configure_deterministic_backend, torch_seed_scope  # noqa: E402


DEFAULT_INPUT = (
    ROOT / "results" / "tip_grid_Supply_Chain_Customer_D_SAITS_Toto2.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "tip_grid_Supply_Chain_Customer_D_SAITS_Toto2_train.json"
)
CKPT_DIR = ROOT / "cache" / "ckpt"
DATASET = "Supply_Chain_Customer_D"
IMPUTER = "SAITS"
PREDICTOR = "Toto2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--eval-batch-size", type=int,
        help="Evaluation batch size; defaults to each tuning run's batch size",
    )
    parser.add_argument(
        "--max-checkpoints", type=int,
        help="Evaluate at most this many pending checkpoints (useful for testing)",
    )
    parser.add_argument("--device", help="Override the configured inference device")
    args = parser.parse_args()
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    if args.max_checkpoints is not None and args.max_checkpoints < 1:
        parser.error("--max-checkpoints must be positive")
    return args


def overrides_for(row: dict) -> list[str]:
    return [
        "--experiment.type", "B2",
        "--data.dataset", DATASET,
        "--imputer.model", IMPUTER,
        "--predictor.model", PREDICTOR,
        "--training.lr", str(row["lr"]),
        "--loader.batch_size", str(row["batch_size"]),
        "--tip.k", str(row["k"]),
        "--tip.epsilon", str(row["epsilon"]),
        "--tip.confidence_delta", str(row["confidence_delta"]),
    ]


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    required = {
        "name", "k", "epsilon", "confidence_delta", "lr", "batch_size"
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
        resolved = resolve_experiment_name(overrides_for(row))
        if resolved != row["name"]:
            raise RuntimeError(
                "The current global/template configuration no longer matches "
                f"the tuning run {row['name']}; resolved name is {resolved}. "
                "Restore the tuning-time non-grid configuration before evaluation."
            )
    return rows


def merge_previous(rows: list[dict], output_path: Path) -> None:
    if not output_path.exists():
        return
    previous = json.loads(output_path.read_text())
    by_name = {row.get("name"): row for row in previous}
    for row in rows:
        old = by_name.get(row["name"])
        if old is not None and old.get("train_pred_mse") is not None:
            row["train_pred_mse"] = old["train_pred_mse"]
            row["train_observed_values"] = old.get("train_observed_values")


def write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_train_loader(dataset: dict, batch_size: int) -> DataLoader:
    train = dataset["train"]
    tensor_dataset = TensorDataset(
        torch.from_numpy(train["X_N"]).float(),
        torch.from_numpy(train["X_NA"]).float(),
        torch.from_numpy(train["X_NA_mask"]).float(),
        torch.from_numpy(train["X_A_mask"]).float(),
        torch.from_numpy(train["Y_N"]).float(),
        torch.from_numpy(train["Y_N_mask"]).float(),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)


@torch.inference_mode()
def compute_train_prediction_mse(
    imputer_model: torch.nn.Module,
    predictor_model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, int]:
    imputer_model.eval()
    predictor_model.eval()
    squared_error = 0.0
    observed_count = 0.0
    for X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask in loader:
        X_N = X_N.to(device)
        X_NA = X_NA.to(device)
        X_NA_mask = X_NA_mask.to(device)
        X_A_mask = X_A_mask.to(device)
        Y_N = Y_N.to(device)
        Y_N_mask = Y_N_mask.to(device)

        imputed = imputer_model(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
        predicted = predictor_model(imputed)["pred_data"]
        observed_mask = 1.0 - Y_N_mask
        squared_error += (
            (predicted - Y_N).square() * observed_mask
        ).sum().item()
        observed_count += observed_mask.sum().item()

    if observed_count == 0:
        raise RuntimeError("Training targets contain no observed future values")
    return squared_error / observed_count, int(observed_count)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("tip-train-prediction")
    configure_deterministic_backend()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    rows = load_rows(input_path)
    merge_previous(rows, output_path)
    pending = [row for row in rows if row.get("train_pred_mse") is None]
    if args.max_checkpoints is not None:
        pending = pending[: args.max_checkpoints]
    if not pending:
        print(f"All requested checkpoints are already evaluated: {output_path}")
        return

    config = config_from_overrides(overrides_for(rows[0]))
    if args.device:
        config.set("training", "device", args.device)
    seed = config.getint("experiment", "seed")
    device = config.get("training", "device")
    cache_dir = Path(config.get("paths", "cache_dir"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    dataset_folder = preprocess_dataset(
        dataset_name=config.get("dataset", "dataset_name"),
        frequency=config.get("dataset", "frequency"),
        item_id=config.get("dataset", "item_id"),
        column_index=config.getint("dataset", "column_index"),
        missing_rate=config.getfloat("data", "missing_rate"),
        artificial_missing_rate=config.getfloat("data", "artificial_missing_rate"),
        block_size=config.getint("data", "block_size"),
        lookback_len=config.getint("data", "lookback"),
        pred_len=config.getint("data", "pred_len"),
        val_ratio=config.getfloat("data", "val_ratio"),
        test_ratio=config.getfloat("data", "test_ratio"),
        random_seed=seed,
        logger=logger,
        save_path=cache_dir / "data",
    )
    dataset = load_h5(dataset_folder / "dataset.h5")
    config.set("data", "d_feature", str(dataset["train"]["X_N"].shape[2]))
    config.set("data", "lookback", str(dataset["train"]["X_N"].shape[1]))
    eval_batch_size = args.eval_batch_size or int(rows[0]["batch_size"])
    loader = build_train_loader(dataset, eval_batch_size)

    with torch_seed_scope(seed, "tip_train_eval_imputer", IMPUTER):
        imputer_model = imputator(IMPUTER, config).to(device).eval()
    with torch_seed_scope(seed, "tip_train_eval_predictor", PREDICTOR):
        predictor_model = predictor(PREDICTOR, config).to(device).eval()

    completed_before = sum(row.get("train_pred_mse") is not None for row in rows)
    total = len(rows)
    print(
        f"Input: {input_path}\nOutput: {output_path}\n"
        f"Checkpoints: {completed_before}/{total} cached, {len(pending)} pending\n"
        f"Training windows: {len(loader.dataset)}, eval_batch_size={eval_batch_size}"
    )

    for offset, row in enumerate(pending, 1):
        checkpoint = CKPT_DIR / f"{row['name']}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        imputer_model.load_state_dict(state)
        mse, observed_values = compute_train_prediction_mse(
            imputer_model, predictor_model, loader, device
        )
        if not math.isfinite(mse):
            raise RuntimeError(f"Non-finite training prediction MSE for {row['name']}")
        row["train_pred_mse"] = mse
        row["train_observed_values"] = observed_values
        write_atomic(output_path, rows)
        print(
            f"[{completed_before + offset:02d}/{total}] "
            f"k={row['k']} eps={row['epsilon']} "
            f"delta={row['confidence_delta']} train_pred_mse={mse:.6f}",
            flush=True,
        )

    print(f"Completed. Results written to: {output_path}")


if __name__ == "__main__":
    main()
