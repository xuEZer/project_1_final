"""Experiment scheduling — phases of imputer training.

Supported experiment types:

- ``B1``: Two-stage baseline — imputer only, reconstruction loss L_I.
- ``B2``: TIP — deterministic local-subspace prediction projection.
"""

from __future__ import annotations

import hashlib
import logging
import math
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

from src.models.jointModel import JointModel


class EpochMonitor:
    """Track the best imputer checkpoint and log losses on train/val splits.

    At every epoch end the monitor computes imputation and prediction MSE
    on the training and validation splits, logs a single concise line, and
    writes scalars to TensorBoard. Supports early stopping.

    Parameters
    ----------
    joint_model : JointModel
    val_loader : DataLoader or None
    num_epoch : int
    patience : int
    min_delta : float
    logger : logging.Logger
    """

    def __init__(
        self,
        joint_model: JointModel,
        val_loader: DataLoader | None,
        num_epoch: int,
        patience: int,
        min_delta: float,
        logger: logging.Logger,
        selection: str = "imputation",
    ) -> None:
        self._model = joint_model
        self._val_loader = val_loader
        self._num_epoch = num_epoch
        self._patience = patience
        self._checkpoint_enabled = patience > 0
        self._min_delta = min_delta
        self._logger = logger
        if selection not in {"imputation", "prediction"}:
            raise ValueError(
                "model selection must be 'imputation' or 'prediction'"
            )
        self._selection = selection
        self._best_val_pred = float("inf")
        self._best_value = float("inf")
        self._best_epoch = 0
        self._best_state: dict[str, torch.Tensor] | None = None
        self._no_improve = 0

    def on_epoch_end(
        self, epoch: int, train_loss: float, model: JointModel
    ) -> bool:
        """Log train/val loss, checkpoint. Returns True if should stop."""
        tb = model.tb_summary_writer

        tb.add_scalar("loss/train_loss", train_loss, epoch)
        parts = [f"train_loss={train_loss:.6f}"]

        val_current = model.compute_val_loss(self._val_loader)
        val_pred_current = (
            model.compute_val_prediction_loss(self._val_loader)
            if (self._val_loader is not None and self._checkpoint_enabled)
            else float("inf")
        )
        if self._val_loader is not None:
            parts.append(f"val_loss={val_current:.6f}")
            tb.add_scalar("loss/val_loss", val_current, epoch)
            if math.isfinite(val_pred_current):
                parts.append(f"val_pred_loss={val_pred_current:.6f}")
                tb.add_scalar("loss/val_pred_loss", val_pred_current, epoch)
        if self._selection == "prediction":
            val_current = val_pred_current

        self._logger.info("Epoch %03d/%d | %s", epoch, self._num_epoch, " | ".join(parts))

        # ``patience <= 0`` is the fixed-horizon mode: validation is still
        # logged, but neither early stopping nor best-checkpoint selection is
        # allowed to replace the final-epoch state.
        if self._selection == "prediction":
            self._best_val_pred = min(self._best_val_pred, val_pred_current)
        if self._checkpoint_enabled and self._val_loader is not None:
            # A non-finite val loss is an invalid observation (e.g. a split
            # with no observed target): neither count it as an improvement
            # nor as a no-improvement, so it cannot trigger early stopping
            # or poison the best checkpoint.
            if math.isfinite(val_current):
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

        if self._checkpoint_enabled and self._no_improve >= self._patience:
            self._logger.info(
                "Early stopping at epoch %03d (no improvement for %d epochs)",
                epoch, self._no_improve,
            )
            return True
        return False

    def restore_best(self) -> None:
        """Load the best imputer state back into the model."""
        if not self._checkpoint_enabled:
            self._logger.info(
                "Early stopping and best-checkpoint selection disabled "
                "(patience=%d); keeping final epoch state.",
                self._patience,
            )
            return
        if self._best_state is None:
            return
        self._model.imputer.load_state_dict(self._best_state)
        self._model.to(self._model.device)
        metric = (
            "val_imp_loss" if self._selection == "imputation"
            else "val_pred_loss"
        )
        self._logger.info(
            "Restored best val checkpoint: epoch=%03d, %s=%.6f",
            self._best_epoch, metric, self._best_value,
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
    patience = config.getint("training", "patience")
    min_delta = config.getfloat("training", "min_delta")

    logger.info("Initial imputer state hash: %s", imputer_state_hash(joint_model))

    epoch_monitor = EpochMonitor(
        joint_model,
        val_loader=train_loader.get("val"),
        num_epoch=num_epoch,
        patience=patience,
        min_delta=min_delta,
        logger=logger,
        selection=config.get(
            "training", "model_selection", fallback="imputation"
        ),
    )
    on_epoch_end = epoch_monitor.on_epoch_end

    if exp_type == "B1":
        trained_model = _run_b1(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "RQ3":
        trained_model = _run_rq3(
            joint_model, train_loader, config, logger, on_epoch_end
        )
    elif exp_type == "RQ4":
        trained_model = _run_rq4(
            joint_model, train_loader, config, logger, on_epoch_end
        )
    elif exp_type in {"B2", "B3", "B4", "B5"}:
        trained_model = _run_b2(
            joint_model, train_loader, config, logger, on_epoch_end,
            exp_type=exp_type,
        )
    else:
        raise ValueError(
            f"Unknown experiment type {exp_type}. "
            "Available types: B1, B2, B3, B4, B5, RQ3, RQ4"
        )

    epoch_monitor.restore_best()
    logger.info("Final imputer state hash: %s", imputer_state_hash(trained_model))

    return trained_model


def imputer_state_hash(joint_model: JointModel) -> str:
    """Return a short deterministic hash of the current imputer weights."""
    digest = hashlib.sha256()
    for name, value in joint_model.imputer.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


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

    def observe_optimizer_step(
        pred_loss_before: float,
        pred_loss_after_raw_adamw: float,
        pred_loss_after_applied_update: float,
    ) -> None:
        joint_model.record_prediction_optimizer_step(
            pred_loss_before=pred_loss_before,
            pred_loss_after_raw_adamw=pred_loss_after_raw_adamw,
            pred_loss_after_applied_update=pred_loss_after_applied_update,
        )

    joint_model.fit(
        data_loader=train_loader,
        phase_name="b1",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        optimizer_step_observer=observe_optimizer_step,
    )

    return joint_model


def _run_b2(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
    exp_type: str = "B2",
) -> JointModel:
    """Run TIP or a matched subspace/correction ablation."""
    num_epoch = config.getint("training", "max_epochs")
    subspace_rank = config.getint("tip", "subspace_rank")
    epsilon = config.getfloat("tip", "epsilon")
    confidence_delta = config.getfloat("tip", "confidence_delta")

    variants = {
        "B2": ("history", "robust", "ucb"),
        "B3": ("random", "robust", "ucb"),
        "B4": ("history", "point", "ucb"),
        "B5": ("history", "robust", "mean"),
    }
    anchor_delta = config.getfloat("tip", "anchor_delta", fallback=0.0)
    basis_mode, correction_mode, gate_mode = variants[exp_type]
    log_phase(
        logger,
        f"{exp_type} — TIP ({basis_mode} subspace, {correction_mode} "
        f"correction, {gate_mode} gate)",
        num_epoch,
    )
    logger.info(
        "  subspace_rank=%d, epsilon=%g, confidence_delta=%g, "
        "basis_mode=%s, correction_mode=%s, gate_mode=%s",
        subspace_rank, epsilon, confidence_delta, basis_mode, correction_mode,
        gate_mode,
    )
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name=exp_type.lower(),
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        subspace_rank=subspace_rank,
        epsilon=epsilon,
        confidence_delta=confidence_delta,
        basis_mode=basis_mode,
        correction_mode=correction_mode,
        gate_mode=gate_mode,
        anchor_delta=anchor_delta,
    )

    return joint_model


def _run_rq3(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
) -> JointModel:
    """Train once with B1 and evaluate B2--B5 counterfactually per step."""
    num_epoch = config.getint("training", "max_epochs")
    subspace_rank = config.getint("tip", "subspace_rank")
    epsilon = config.getfloat("tip", "epsilon")
    confidence_delta = config.getfloat("tip", "confidence_delta")
    log_phase(logger, "RQ3 — shared B1 trajectory", num_epoch)
    joint_model.fit_rq3_counterfactual(
        data_loader=train_loader,
        phase_name="rq3",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        subspace_rank=subspace_rank,
        epsilon=epsilon,
        confidence_delta=confidence_delta,
    )
    return joint_model


def _csv_values(config: ConfigParser, key: str, cast):
    values = [
        cast(value.strip())
        for value in config.get("tip", key).split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError(f"tip.{key} must contain at least one value")
    return values


def _run_rq4(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
) -> JointModel:
    """Train B1 once and evaluate all RQ4 sensitivity candidates per step."""
    num_epoch = config.getint("training", "max_epochs")
    base_rank = config.getint("tip", "subspace_rank")
    base_epsilon = config.getfloat("tip", "epsilon")
    base_delta = config.getfloat("tip", "confidence_delta")
    ranks = _csv_values(config, "rq4_ranks", int)
    epsilons = _csv_values(config, "rq4_epsilons", float)
    gate_deltas = _csv_values(config, "rq4_gate_deltas", float)
    log_phase(logger, "RQ4 — shared B1 sensitivity trajectory", num_epoch)
    joint_model.fit_rq4_counterfactual(
        data_loader=train_loader,
        phase_name="rq4",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        base_rank=base_rank,
        base_epsilon=base_epsilon,
        cone_delta=base_delta,
        ranks=ranks,
        epsilons=epsilons,
        gate_deltas=gate_deltas,
    )
    return joint_model


def log_phase(logger: logging.Logger, phase: str, epochs: int) -> None:
    """Log a phase boundary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase: {phase} ({epochs} max epochs)")
    logger.info("=" * 60)
