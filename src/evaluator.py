"""Model evaluation utilities for imputation and forecasting tasks."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

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


def evaluate(
    config: ConfigParser,
    best_joint_model: JointModel,
    eval_loader: DataLoader,
    no_gt: bool,
    logger: logging.Logger,
    split_name: str = "test",
    run_predictor: bool = True,
) -> dict[str, float | None]:
    """Evaluate a split, optionally skipping forecast metrics."""
    logger.info("Starting evaluation on %s split...", split_name)

    totals = {
        kind: {"sq_error": 0.0, "count": 0.0}
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
        totals[kind]["sq_error"] += (error.square() * mask).sum().item()
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
            outputs = best_joint_model.forward(
                X_N, X_N_mask, run_predictor=run_predictor
            )
            imputed = outputs["imp_data"]

            if X is not None and X_mask is not None:
                accumulate("imp", imputed, X, X_mask)
            if run_predictor:
                pred = _require_prediction(outputs, "evaluation")
                accumulate("pred", pred, Y, Y_mask)

    logger.info("-" * 50)
    logger.info("Evaluation Results Summary: split=%s", split_name)
    logger.info(f"{'':<10} {'Imputation':<20} {'Prediction':<20}")
    logger.info("-" * 50)

    def finalize(kind: str) -> dict[str, float] | None:
        values = totals[kind]
        if values["count"] <= 0:
            return None
        return {"mse": values["sq_error"] / values["count"]}

    imp_metrics = finalize("imp")
    pred_metrics = finalize("pred") if run_predictor else None
    if run_predictor and pred_metrics is None:
        # A split can have no observed prediction targets (all masks missing);
        # that is a valid outcome, not an error — report pred_* as N/A.
        logger.info(
            "No observed prediction targets on %s split — pred_* metrics are N/A.",
            split_name,
        )

    def _fmt(value: float | None) -> str:
        return f"{value:<20.6f}" if value is not None else f"{'N/A':<20}"

    metrics: dict[str, float | None] = {}
    for name in ("mse",):
        avg_imp = imp_metrics[name] if imp_metrics is not None else None
        avg_pred = pred_metrics[name] if pred_metrics is not None else None
        metrics[f"imp_{name}"] = avg_imp
        if run_predictor:
            metrics[f"pred_{name}"] = avg_pred
        logger.info(f"{name.upper():<10} {_fmt(avg_imp)} {_fmt(avg_pred)}")

    logger.info("-" * 50)
    logger.info("Evaluation completed on %s split.", split_name)
    return metrics
