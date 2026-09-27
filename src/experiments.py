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

from torch.utils.data import DataLoader

from src.models.jointModel import JointModel


class EpochMonitor:
    """Log training imputation and validation losses after each epoch."""

    def __init__(
        self,
        joint_model: JointModel,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        num_epoch: int,
        logger: logging.Logger,
    ) -> None:
        self._model = joint_model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._num_epoch = num_epoch
        self._logger = logger
        self.last_metrics: dict[str, float] = {}

    def on_epoch_end(
        self, epoch: int, train_loss: float, model: JointModel
    ) -> None:
        """Log reconstruction loss and validation forecast loss."""
        train_imp = (
            model.compute_val_loss(self._train_loader)
            if self._train_loader is not None else float("inf")
        )
        val_imp = (
            model.compute_val_loss(self._val_loader)
            if self._val_loader is not None else float("inf")
        )
        val_pred = (
            model.compute_val_prediction_loss(
                self._val_loader,
                rng_namespace="epoch_prediction_monitor_validation",
                rng_step=epoch,
            )
            if self._val_loader is not None else float("inf")
        )

        parts = [f"train_loss={train_loss:.6f}"]
        for name, value in (
            ("train_imp_loss", train_imp),
            ("val_imp_loss", val_imp),
            ("val_pred_loss", val_pred),
        ):
            if math.isfinite(value):
                parts.append(f"{name}={value:.6f}")
        self._logger.info("Epoch %03d/%d | %s", epoch, self._num_epoch, " | ".join(parts))

        self.last_metrics = {
            "train_imp_loss": train_imp,
            "val_imp_loss": val_imp,
            "val_pred_loss": val_pred,
        }


def run_experiment(
    exp_type: str,
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch=None,
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
        The trained joint model, with the final-epoch imputer delivered
        directly: no checkpoint selection, so endpoint metrics reflect the
        model that is actually shipped rather than a single-epoch minimum of
        a noisy validation forecast loss.

    Raises:
        ValueError: If ``exp_type`` is not recognised.
    """
    num_epoch = config.getint("training", "max_epochs")

    logger.info("Initial imputer state hash: %s", imputer_state_hash(joint_model))

    epoch_monitor = EpochMonitor(
        joint_model,
        train_loader=train_loader.get("train_eval"),
        val_loader=train_loader.get("val"),
        num_epoch=num_epoch,
        logger=logger,
    )
    base_on_epoch_end = epoch_monitor.on_epoch_end
    final_epoch: int | None = None
    final_val_pred: float | None = None

    def on_epoch_end(epoch, train_loss, model):
        """Monitor each epoch, then notify; the final epoch is delivered."""
        nonlocal final_epoch, final_val_pred
        base_on_epoch_end(epoch, train_loss, model)
        val_pred = epoch_monitor.last_metrics.get("val_pred_loss")
        if val_pred is not None and math.isfinite(val_pred):
            final_epoch = epoch
            final_val_pred = val_pred
        if on_epoch is not None:
            on_epoch(epoch, epoch_monitor)

    if exp_type == "B1":
        trained_model = _run_b1(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type in {"B2", "B3", "B4", "B5", "B6"}:
        trained_model = _run_b2(
            joint_model, train_loader, config, logger, on_epoch_end,
            exp_type=exp_type,
        )
    else:
        raise ValueError(
            f"Unknown experiment type {exp_type}. "
            "Available types: B1, B2, B3, B4, B5, B6"
        )

    if final_epoch is not None and final_val_pred is not None:
        trained_model.record_metric(
            "selection/best_epoch", float(final_epoch), final_epoch
        )
        trained_model.record_metric(
            "selection/best_val_pred_loss", final_val_pred, final_epoch
        )
        logger.info(
            "Delivering final-epoch imputer directly: epoch=%d | "
            "val_pred_loss=%.9g",
            final_epoch,
            final_val_pred,
        )
    else:
        logger.warning(
            "No finite validation prediction loss was observed; retaining "
            "the final-epoch imputer"
        )

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
    """B1: Train the imputer with reconstruction loss."""
    num_epoch = config.getint("training", "max_epochs")

    log_phase(logger, "B1 — Standard Imputer Training", num_epoch)

    joint_model.fit(
        data_loader=train_loader,
        phase_name="b1",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
    )

    return joint_model


def _run_b2(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
    exp_type: str = "B2",
    initial_history=None,
) -> JointModel:
    """Run PILOT or a matched subspace/correction ablation."""
    num_epoch = config.getint("training", "max_epochs")
    subspace_rank = config.getint("pilot", "subspace_rank")
    epsilon = config.getfloat("pilot", "epsilon")
    # A single confidence level drives both the release test and the robust
    # projection.
    delta = config.getfloat("pilot", "delta", fallback=0.3)
    cone_radius = config.get("pilot", "cone_radius", fallback="hotelling")
    finite_difference_third_derivative = config.getfloat(
        "pilot", "finite_difference_third_derivative", fallback=0.0,
    )
    # Fairness control: give B2 the same displacement norm as B1 so the two
    # arms differ only in the direction of the step.
    preserve_norm = config.getboolean("pilot", "preserve_norm", fallback=False)

    robust_gate_mode = config.get("pilot", "gate_mode", fallback="safe_release")
    # B6: the fired correction is replaced by the estimated forecast-gradient
    # direction itself (norm-preserved), the direct "descend Q" control.
    variants = {
        "B2": ("history", "robust", robust_gate_mode),
        "B3": ("random", "robust", robust_gate_mode),
        "B4": ("history", "point", robust_gate_mode),
        "B5": ("history", "robust", "mean"),
        "B6": ("history", "gradient", robust_gate_mode),
    }
    prediction_probe_batches = config.getint(
        "pilot", "prediction_probe_batches", fallback=1,
    )
    basis_mode, correction_mode, gate_mode = variants[exp_type]
    log_phase(
        logger,
        f"{exp_type} — PILOT ({basis_mode} subspace, {correction_mode} "
        f"correction, {gate_mode} gate)",
        num_epoch,
    )
    logger.info(
        "  subspace_rank=%d, epsilon=%g, delta=%g, "
        "cone_radius=%s, fd_third_derivative=%g, preserve_norm=%s, "
        "basis_mode=%s, correction_mode=%s, gate_mode=%s",
        subspace_rank, epsilon, delta, cone_radius,
        finite_difference_third_derivative, preserve_norm,
        basis_mode, correction_mode, gate_mode,
    )
    joint_model.fit_pilot(
        data_loader=train_loader,
        phase_name=exp_type.lower(),
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        subspace_rank=subspace_rank,
        epsilon=epsilon,
        gate_delta=delta,
        cone_delta=delta,
        cone_radius=cone_radius,
        finite_difference_third_derivative=finite_difference_third_derivative,
        preserve_norm=preserve_norm,
        basis_mode=basis_mode,
        correction_mode=correction_mode,
        gate_mode=gate_mode,
        prediction_probe_batches=prediction_probe_batches,
        initial_history=initial_history,
    )

    return joint_model

def log_phase(logger: logging.Logger, phase: str, epochs: int) -> None:
    """Log a phase boundary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase: {phase} ({epochs} max epochs)")
    logger.info("=" * 60)
