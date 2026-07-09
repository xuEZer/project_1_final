"""Experiment scheduling — phases of imputer training.

Supported experiment types:

- ``B1``: Two-stage baseline — imputer only, reconstruction loss L_I.
- ``B2``: TIP — zero-order ĝ_P + conditional orthogonal projection.
"""

from __future__ import annotations

import hashlib
import logging
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

from src.evaluator import compute_all_losses
from src.models.jointModel import JointModel


class EpochMonitor:
    """Track the best imputer checkpoint and log losses on train/val splits.

    At every epoch end the monitor computes imputation and prediction MSE
    on the training and validation splits, logs a single concise line, and
    writes scalars to TensorBoard. Supports early stopping.

    Parameters
    ----------
    joint_model : JointModel
    train_loader : DataLoader
    val_loader : DataLoader or None
    patience : int
        Early stopping patience (epochs without improvement).
    min_delta : float
        Minimum improvement in val imp_loss to count as a new best.
    logger : logging.Logger
    """

    def __init__(
        self,
        joint_model: JointModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        patience: int,
        min_delta: float,
        logger: logging.Logger,
    ) -> None:
        self._model = joint_model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._patience = patience
        self._min_delta = min_delta
        self._logger = logger
        self._best_value = float("inf")
        self._best_epoch = 0
        self._best_state: dict[str, torch.Tensor] | None = None
        self._no_improve = 0

    def on_epoch_end(
        self, epoch: int, _train_loss: float, model: JointModel
    ) -> bool:
        """Evaluate splits, log, checkpoint. Returns True if should stop."""
        tb = model.tb_summary_writer

        parts: list[str] = []
        val_current = float("inf")

        for split_name, loader in [
            ("train", self._train_loader),
            ("val", self._val_loader),
        ]:
            if loader is None:
                continue
            losses = compute_all_losses(model, loader)
            imp = losses["imp_loss"]
            pred = losses["pred_loss"]
            parts.append(f"{split_name}_imp={imp:.6f} {split_name}_pred={pred:.6f}")
            tb.add_scalar(f"loss/{split_name}_imp", imp, epoch)
            tb.add_scalar(f"loss/{split_name}_pred", pred, epoch)
            if split_name == "val":
                val_current = losses["imp_loss"]

        self._logger.info("Epoch %03d | %s", epoch, " | ".join(parts))

        # Checkpoint & early stopping (always on val imp_loss)
        if self._val_loader is not None:
            if self._best_value - val_current > self._min_delta:
                self._best_value = val_current
                self._best_epoch = epoch
                self._best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.imputer.state_dict().items()
                }
                self._no_improve = 0
            else:
                self._no_improve += 1

        if self._no_improve >= self._patience:
            self._logger.info(
                "Early stopping at epoch %03d (no improvement for %d epochs)",
                epoch, self._no_improve,
            )
            return True
        return False

    # ------------------------------------------------------------------
    def restore_best(self) -> None:
        """Load the best imputer state back into the model."""
        if self._best_state is None:
            return
        self._model.imputer.load_state_dict(self._best_state)
        self._model.to(self._model.device)
        self._logger.info(
            "Restored best val checkpoint: epoch=%03d, imp_loss=%.6f",
            self._best_epoch, self._best_value,
        )


def run_experiment(
    exp_type: str,
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """Run the configured experiment type.

    Args:
        exp_type: Experiment type identifier (``"B1"`` or ``"B2"``).
        joint_model: Initialised joint model (imputer + frozen predictor).
        train_loader: Dictionary with ``"train"`` DataLoader and
            optionally ``"val"``.
        config: Parsed experiment configuration.
        logger: Logger instance.

    Returns:
        The trained joint model after the final epoch.

    Raises:
        ValueError: If ``exp_type`` is not recognised.
    """
    num_epoch = config.getint("training", "max_epochs")
    patience = config.getint("training", "patience", fallback=20)
    min_delta = config.getfloat("training", "min_delta", fallback=1e-4)

    logger.info("Initial imputer state hash: %s", imputer_state_hash(joint_model))

    epoch_monitor = EpochMonitor(
        joint_model,
        train_loader=train_loader.get("train"),
        val_loader=train_loader.get("val"),
        patience=patience,
        min_delta=min_delta,
        logger=logger,
    )
    on_epoch_end = epoch_monitor.on_epoch_end

    if exp_type == "B1":
        trained_model = _run_b1(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B2":
        trained_model = _run_b2(joint_model, train_loader, config, logger, on_epoch_end)
    else:
        raise ValueError(
            f"Unknown experiment type {exp_type}. Available types: B1, B2"
        )

    epoch_monitor.restore_best()

    return trained_model


def imputer_state_hash(joint_model: JointModel) -> str:
    """Return a short deterministic hash of the current imputer weights."""
    digest = hashlib.sha256()
    for name, value in joint_model.imputer.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


# ------------------------------------------------------------------ B1 --


def _run_b1(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
) -> JointModel:
    """B1: Two-stage baseline — train imputer only with reconstruction loss."""
    num_epoch = config.getint("training", "max_epochs")

    log_phase(logger, "B1 — Standard Imputer Training", num_epoch)
    joint_model.fit(
        data_loader=train_loader,
        phase_name="b1",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
    )

    return joint_model


# ------------------------------------------------------------------ B2 --


def _run_b2(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
) -> JointModel:
    """B2: Full TIP — zero-order ĝ_P estimation + conditional projection."""
    num_epoch = config.getint("training", "max_epochs")
    k = config.getint("tip", "k", fallback=20)
    epsilon = config.getfloat("tip", "epsilon", fallback=1e-4)

    log_phase(logger, "B2 — TIP Training", num_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b2",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        k=k,
        epsilon=epsilon,
    )

    return joint_model


# ------------------------------------------------------------------ util --


def log_phase(logger: logging.Logger, phase: str, epochs: int) -> None:
    """Log a phase boundary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase: {phase} ({epochs} max epochs)")
    logger.info("=" * 60)
