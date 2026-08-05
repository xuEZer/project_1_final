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
    """Yield normalized batches for training-format validation and test loaders."""
    for batch in eval_loader:
        tensors = joint_model._move_batch_to_device(batch)
        if len(tensors) == 6:
            X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = tensors
            yield X_NA, X_NA_mask, X_N, X_A_mask, Y_N, 1 - Y_N_mask
        elif no_gt:
            X_N, X_N_mask, Y_N, Y_N_mask = tensors
            yield X_N, X_N_mask, None, None, Y_N, 1 - Y_N_mask
        else:
            X_N, X_N_mask, X, Y = tensors
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
    run_predictor: bool = True,
) -> dict[str, float]:
    """Compute averaged imputation (and optionally prediction) MSE losses.

    Args:
        joint_model: Joint model to evaluate.
        eval_loader: Dataloader (training or evaluation format).
        run_predictor: If False, skip prediction and only compute imp_loss.

    Returns:
        ``{"imp_loss": float, "pred_loss": float}``.
    """
    imp_total = 0.0
    imp_weight = 0.0
    pred_total = 0.0
    pred_weight = 0.0
    no_gt = getattr(eval_loader, "no_gt", False)
    with _evaluation_state(joint_model), torch.no_grad():
        for batch in eval_loader:
            tensors = joint_model._move_batch_to_device(batch)
            if len(tensors) == 6:
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = tensors
                outputs = joint_model.forward(
                    X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                    run_predictor=run_predictor,
                )
                imputed = outputs["imp_data"]
                pred = outputs.get("pred_data")

                w = X_A_mask.sum().item()
                imp_total += masked_mse_cal(imputed, X_N, X_A_mask).item() * w
                imp_weight += w

                if run_predictor and pred is not None:
                    pred_mask = 1.0 - Y_N_mask
                    w = pred_mask.sum().item()
                    pred_total += masked_mse_cal(pred, Y_N, pred_mask).item() * w
                    pred_weight += w
            elif no_gt:
                if not run_predictor:
                    continue
                X_N, X_N_mask, Y_N, Y_N_mask = tensors
                outputs = joint_model.forward(X_N, X_N_mask)
                pred = outputs.get("pred_data")

                if pred is not None:
                    pred_mask = 1.0 - Y_N_mask
                    w = pred_mask.sum().item()
                    pred_total += masked_mse_cal(pred, Y_N, pred_mask).item() * w
                    pred_weight += w
            else:
                X_N, X_N_mask, X, Y = tensors
                outputs = joint_model.forward(
                    X_N, X_N_mask, X, X_N_mask, Y, torch.ones_like(Y),
                    run_predictor=run_predictor,
                )
                imputed = outputs["imp_data"]
                pred = outputs.get("pred_data")

                w = X_N_mask.sum().item()
                imp_total += masked_mse_cal(imputed, X, X_N_mask).item() * w
                imp_weight += w

                if run_predictor and pred is not None:
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

    totals = {
        kind: {"abs_error": 0.0, "sq_error": 0.0, "target_abs": 0.0, "count": 0.0}
        for kind in ("imp", "pred")
    }

    def accumulate(
        kind: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Accumulate sufficient statistics for exact dataset-level metrics."""
        error = prediction - target
        totals[kind]["abs_error"] += (error.abs() * mask).sum().item()
        totals[kind]["sq_error"] += (error.square() * mask).sum().item()
        totals[kind]["target_abs"] += (target.abs() * mask).sum().item()
        totals[kind]["count"] += mask.sum().item()

    if no_gt:
        logger.info(
            "No ground truth for natural missing positions. "
            "Imputation metrics will use artificial-mask ground truth when available."
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

            if X is not None and X_mask is not None:
                accumulate("imp", imputed, X, X_mask)
            accumulate("pred", pred, Y, Y_mask)

    logger.info("-" * 50)
    logger.info("Evaluation Results Summary: split=%s", split_name)
    logger.info(f"{'':<10} {'Imputation':<20} {'Prediction':<20}")
    logger.info("-" * 50)

    def finalize(kind: str) -> dict[str, float] | None:
        values = totals[kind]
        if values["count"] <= 0:
            return None
        mse = values["sq_error"] / values["count"]
        return {
            "mae": values["abs_error"] / values["count"],
            "mse": mse,
            "mre": values["abs_error"] / (values["target_abs"] + 1e-9),
            "rmse": mse ** 0.5,
        }

    imp_metrics = finalize("imp")
    pred_metrics = finalize("pred")
    if pred_metrics is None:
        # A split can have no observed prediction targets (all masks missing);
        # that is a valid outcome, not an error — report pred_* as N/A.
        logger.info(
            "No observed prediction targets on %s split — pred_* metrics are N/A.",
            split_name,
        )

    def _fmt(value: float | None) -> str:
        return f"{value:<20.6f}" if value is not None else f"{'N/A':<20}"

    metrics: dict[str, float | None] = {}
    for name in ("mae", "mse", "mre", "rmse"):
        avg_imp = imp_metrics[name] if imp_metrics is not None else None
        avg_pred = pred_metrics[name] if pred_metrics is not None else None
        metrics[f"imp_{name}"] = avg_imp
        metrics[f"pred_{name}"] = avg_pred
        logger.info(f"{name.upper():<10} {_fmt(avg_imp)} {_fmt(avg_pred)}")

    logger.info("-" * 50)
    logger.info("Evaluation completed on %s split.", split_name)
    return metrics
