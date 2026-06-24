"""Model evaluation utilities for imputation and forecasting tasks."""

import logging
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


def evaluate(
    config: ConfigParser,
    best_joint_model: JointModel,
    eval_loader: DataLoader,
    no_gt: bool,
    logger: logging.Logger,
) -> None:
    """Evaluate a trained joint model on the evaluation dataloader.

    Args:
        config: Experiment configuration with runtime options.
        best_joint_model: Trained joint model used for evaluation.
        eval_loader: Evaluation DataLoader (not a dict).
        no_gt: Whether test split lacks complete ground truth for imputation.
        logger: Logger for evaluation progress and metrics.

    Returns:
        None.

    """
    logger.info("Starting evaluation...")

    device = config.get("training", "device")

    best_joint_model.to(device)
    best_joint_model.eval()

    mae_stats = {"imp": 0.0, "pred": 0.0, "count": 0}
    mse_stats = {"imp": 0.0, "pred": 0.0, "count": 0}
    mre_stats = {"imp": 0.0, "pred": 0.0, "count": 0}
    rmse_stats = {"imp": 0.0, "pred": 0.0, "count": 0}

    with torch.no_grad():
        if no_gt:
            logger.info(
                "No ground truth for natural missing positions. "
                "Skipping imputation evaluation."
            )
            for batch in eval_loader:
                X_N, X_N_mask, Y_N, Y_N_mask = [b.to(device) for b in batch]

                outputs = best_joint_model.forward(X_N, X_N_mask)
                pred = outputs["pred_data"]

                pred_mae = masked_mae_cal(pred, Y_N, 1 - Y_N_mask)
                pred_mse = masked_mse_cal(pred, Y_N, 1 - Y_N_mask)
                pred_mre = masked_mre_cal(pred, Y_N, 1 - Y_N_mask)
                pred_rmse = masked_rmse_cal(pred, Y_N, 1 - Y_N_mask)

                mae_stats["pred"] += pred_mae.item()
                mae_stats["count"] += 1
                mse_stats["pred"] += pred_mse.item()
                mse_stats["count"] += 1
                mre_stats["pred"] += pred_mre.item()
                mre_stats["count"] += 1
                rmse_stats["pred"] += pred_rmse.item()
                rmse_stats["count"] += 1

        else:
            for batch in eval_loader:
                X_N, X_N_mask, X, Y = [b.to(device) for b in batch]

                outputs = best_joint_model.forward(X_N, X_N_mask)

                imputed = outputs["imp_data"]
                pred = outputs["pred_data"]

                imp_mae = masked_mae_cal(imputed, X, X_N_mask)
                imp_mse = masked_mse_cal(imputed, X, X_N_mask)
                imp_mre = masked_mre_cal(imputed, X, X_N_mask)
                imp_rmse = masked_rmse_cal(imputed, X, X_N_mask)
                pred_mae = masked_mae_cal(pred, Y, torch.ones_like(Y))
                pred_mse = masked_mse_cal(pred, Y, torch.ones_like(Y))
                pred_mre = masked_mre_cal(pred, Y, torch.ones_like(Y))
                pred_rmse = masked_rmse_cal(pred, Y, torch.ones_like(Y))

                mae_stats["imp"] += imp_mae.item()
                mae_stats["pred"] += pred_mae.item()
                mae_stats["count"] += 1
                mse_stats["imp"] += imp_mse.item()
                mse_stats["pred"] += pred_mse.item()
                mse_stats["count"] += 1
                mre_stats["imp"] += imp_mre.item()
                mre_stats["pred"] += pred_mre.item()
                mre_stats["count"] += 1
                rmse_stats["imp"] += imp_rmse.item()
                rmse_stats["pred"] += pred_rmse.item()
                rmse_stats["count"] += 1

    logger.info("-" * 50)
    logger.info("Evaluation Results Summary:")
    if no_gt:
        logger.info(f"{'':<10} {'Prediction':<20}")
    else:
        logger.info(f"{'':<10} {'Imputation':<20} {'Prediction':<20}")
    logger.info("-" * 50)

    count = max(1, mae_stats["count"])
    avg_pred_mae = mae_stats["pred"] / count
    avg_pred_mse = mse_stats["pred"] / count
    avg_pred_mre = mre_stats["pred"] / count
    avg_pred_rmse = rmse_stats["pred"] / count

    if no_gt:
        logger.info(f"{'MAE':<10} {'N/A':<20} {avg_pred_mae:<20.6f}")
        logger.info(f"{'MSE':<10} {'N/A':<20} {avg_pred_mse:<20.6f}")
        logger.info(f"{'MRE':<10} {'N/A':<20} {avg_pred_mre:<20.6f}")
        logger.info(f"{'RMSE':<10} {'N/A':<20} {avg_pred_rmse:<20.6f}")
    else:
        avg_imp_mae = mae_stats["imp"] / count
        avg_imp_mse = mse_stats["imp"] / count
        avg_imp_mre = mre_stats["imp"] / count
        avg_imp_rmse = rmse_stats["imp"] / count
        logger.info(f"{'MAE':<10} {avg_imp_mae:<20.6f} {avg_pred_mae:<20.6f}")
        logger.info(f"{'MSE':<10} {avg_imp_mse:<20.6f} {avg_pred_mse:<20.6f}")
        logger.info(f"{'MRE':<10} {avg_imp_mre:<20.6f} {avg_pred_mre:<20.6f}")
        logger.info(f"{'RMSE':<10} {avg_imp_rmse:<20.6f} {avg_pred_rmse:<20.6f}")

    logger.info("-" * 50)
    logger.info("Evaluation completed.")
