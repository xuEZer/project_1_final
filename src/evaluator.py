"""Model evaluation utilities for imputation and forecasting tasks."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

from src.models.extern.SAITS.utils import (
    masked_mae_cal,
    masked_mre_cal,
    masked_mse_cal,
    masked_rmse_cal,
)
from src.models.jointModel import JointModel


@contextlib.contextmanager
def _evaluation_state(
    joint_model: JointModel,
    device: str | None = None,
    restore_training: bool = True,
) -> Iterator[None]:
    """Switch a joint model to eval mode and restore training state."""
    was_training = joint_model.training
    joint_model.to(device or joint_model.device)
    joint_model.eval()
    try:
        yield
    finally:
        if restore_training and was_training:
            joint_model.train()
            joint_model.predictor.eval()


def _iter_eval_batches(
    joint_model: JointModel,
    eval_loader: DataLoader,
    no_gt: bool,
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """Yield normalized eval batches for both public and complete test splits."""
    if no_gt:
        for batch in eval_loader:
            X_N, X_N_mask, Y_N, Y_N_mask = joint_model._move_batch_to_device(batch)
            yield X_N, X_N_mask, None, None, Y_N, 1 - Y_N_mask
    else:
        for batch in eval_loader:
            X_N, X_N_mask, X, Y = joint_model._move_batch_to_device(batch)
            yield X_N, X_N_mask, X, X_N_mask, Y, torch.ones_like(Y)


def _require_prediction(
    outputs: dict[str, torch.Tensor | None],
    stage: str,
) -> torch.Tensor:
    pred = outputs.get("pred_data")
    if pred is None:
        raise RuntimeError(
            f"Predictor returned None during {stage}. "
            "Ensure a real predictor is loaded before evaluation."
        )
    return pred


def compute_prediction_mae(
    joint_model: JointModel,
    eval_loader: DataLoader,
) -> float:
    """Compute average prediction MAE on a dataloader."""
    total_mae = 0.0
    total_weight = 0.0
    with _evaluation_state(joint_model), torch.no_grad():
        for X_N, X_N_mask, _X, _X_mask, Y, Y_mask in _iter_eval_batches(
            joint_model, eval_loader, False,
        ):
            outputs = joint_model.forward(X_N, X_N_mask)
            pred = _require_prediction(outputs, "validation")
            w = Y_mask.sum().item()
            total_mae += masked_mae_cal(pred, Y, Y_mask).item() * w
            total_weight += w
    return total_mae / max(1.0, total_weight)


def compute_eval_losses(
    joint_model: JointModel,
    eval_loader: DataLoader,
) -> dict[str, float | None]:
    """Compute averaged imputation MSE loss on a dataloader."""
    result = compute_all_losses(joint_model, eval_loader)
    return {"imp_loss": result["imp_loss"]}


def compute_all_losses(
    joint_model: JointModel,
    eval_loader: DataLoader,
) -> dict[str, float]:
    """Compute averaged imputation and prediction MSE losses on a dataloader.

    Handles both training batches (6 tensors:
    ``X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask``) and evaluation
    batches (4 tensors: ``X_N, X_N_mask, X, Y``).

    Returns:
        ``{"imp_loss": float, "pred_loss": float}``.  Either value may be
        ``float("nan")`` if the loader does not provide ground truth for
        that task.
    """
    imp_total = 0.0
    imp_weight = 0.0
    pred_total = 0.0
    pred_weight = 0.0
    with _evaluation_state(joint_model), torch.no_grad():
        for batch in eval_loader:
            tensors = joint_model._move_batch_to_device(batch)
            if len(tensors) == 6:
                # Training format
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = tensors
                outputs = joint_model.forward(
                    X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                )
                imputed = outputs["imp_data"]
                pred = outputs.get("pred_data")

                w = X_A_mask.sum().item()
                imp_total += masked_mse_cal(imputed, X_N, X_A_mask).item() * w
                imp_weight += w

                if pred is not None:
                    pred_mask = 1.0 - Y_N_mask
                    w = pred_mask.sum().item()
                    pred_total += masked_mse_cal(pred, Y_N, pred_mask).item() * w
                    pred_weight += w
            else:
                # Evaluation format
                X_N, X_N_mask, X, Y = tensors
                outputs = joint_model.forward(
                    X_N, X_N_mask, X, X_N_mask, Y, torch.ones_like(Y),
                )
                imputed = outputs["imp_data"]
                pred = outputs.get("pred_data")

                w = X_N_mask.sum().item()
                imp_total += masked_mse_cal(imputed, X, X_N_mask).item() * w
                imp_weight += w

                if pred is not None:
                    pred_mask = torch.ones_like(Y)
                    w = pred_mask.sum().item()
                    pred_total += masked_mse_cal(pred, Y, pred_mask).item() * w
                    pred_weight += w

    return {
        "imp_loss": imp_total / imp_weight if imp_weight else float("nan"),
        "pred_loss": pred_total / pred_weight if pred_weight else float("nan"),
    }


def evaluate(
    config: ConfigParser,
    best_joint_model: JointModel,
    eval_loader: DataLoader,
    no_gt: bool,
    logger: logging.Logger,
    split_name: str = "test",
) -> dict[str, float | None]:
    """Evaluate a trained joint model on the evaluation dataloader."""
    logger.info("Starting evaluation on %s split...", split_name)

    metric_fns = {
        "mae": masked_mae_cal,
        "mse": masked_mse_cal,
        "mre": masked_mre_cal,
        "rmse": masked_rmse_cal,
    }
    totals = {
        name: {"imp": 0.0, "pred": 0.0, "imp_weight": 0.0, "pred_weight": 0.0}
        for name in metric_fns
    }

    if no_gt:
        logger.info(
            "No ground truth for natural missing positions. "
            "Skipping imputation evaluation."
        )

    device = config.get("training", "device")
    with _evaluation_state(
        best_joint_model, device=device, restore_training=False
    ), torch.no_grad():
        for X_N, X_N_mask, X, X_mask, Y, Y_mask in _iter_eval_batches(
            best_joint_model, eval_loader, no_gt
        ):
            outputs = best_joint_model.forward(X_N, X_N_mask)
            imputed = outputs["imp_data"]
            pred = _require_prediction(outputs, "evaluation")

            for name, metric_fn in metric_fns.items():
                if X is not None and X_mask is not None:
                    w = X_mask.sum().item()
                    totals[name]["imp"] += metric_fn(imputed, X, X_mask).item() * w
                    totals[name]["imp_weight"] += w
                w = Y_mask.sum().item()
                totals[name]["pred"] += metric_fn(pred, Y, Y_mask).item() * w
                totals[name]["pred_weight"] += w

    logger.info("-" * 50)
    logger.info("Evaluation Results Summary: split=%s", split_name)
    if no_gt:
        logger.info(f"{'':<10} {'Prediction':<20}")
    else:
        logger.info(f"{'':<10} {'Imputation':<20} {'Prediction':<20}")
    logger.info("-" * 50)

    metrics: dict[str, float | None] = {}
    for name in ("mae", "mse", "mre", "rmse"):
        imp_w = max(1.0, totals[name]["imp_weight"])
        pred_w = max(1.0, totals[name]["pred_weight"])
        avg_imp = None if no_gt else totals[name]["imp"] / imp_w
        avg_pred = totals[name]["pred"] / pred_w
        metrics[f"imp_{name}"] = avg_imp
        metrics[f"pred_{name}"] = avg_pred
        if no_gt:
            logger.info(f"{name.upper():<10} {'N/A':<20} {avg_pred:<20.6f}")
        else:
            logger.info(f"{name.upper():<10} {avg_imp:<20.6f} {avg_pred:<20.6f}")

    logger.info("-" * 50)
    logger.info("Evaluation completed on %s split.", split_name)
    return metrics
