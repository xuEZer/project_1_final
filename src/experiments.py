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
    """Log the imputation and prediction loss on both splits every epoch.

    Training runs to a fixed horizon: there is no early stopping and no
    best-checkpoint selection, so the state left by the last epoch is the one
    that gets evaluated.  Both splits are measured with the same pair of
    helpers on the same training-format loaders, so ``train_*`` and ``val_*``
    are directly comparable; the loader also carries the artificial masks that
    make the imputation loss well defined.

    Parameters
    ----------
    joint_model : JointModel
    train_loader : DataLoader or None
        Unshuffled view of the training split, in the training tensor layout.
    val_loader : DataLoader or None
    num_epoch : int
    logger : logging.Logger
    """

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
        # Last epoch's metrics, for the observer that forwards them to expman.
        self.last_metrics: dict[str, float | None] = {}

    @staticmethod
    def _losses(model: JointModel, loader: DataLoader | None) -> tuple[float, float]:
        """Imputation and prediction loss on one split, same convention for both."""
        if loader is None:
            return float("inf"), float("inf")
        return (
            model.compute_val_loss(loader),
            model.compute_val_prediction_loss(loader),
        )

    def on_epoch_end(
        self, epoch: int, train_loss: float, model: JointModel
    ) -> None:
        """Log the four losses for one finished epoch."""
        train_imp, train_pred = self._losses(model, self._train_loader)
        val_imp, val_pred = self._losses(model, self._val_loader)

        parts = [f"train_loss={train_loss:.6f}"]
        for name, value in (
            ("train_imp_loss", train_imp),
            ("train_pred_loss", train_pred),
            ("val_imp_loss", val_imp),
            ("val_pred_loss", val_pred),
        ):
            if math.isfinite(value):
                parts.append(f"{name}={value:.6f}")
        self._logger.info("Epoch %03d/%d | %s", epoch, self._num_epoch, " | ".join(parts))

        self.last_metrics = {
            "train_imp_loss": train_imp,
            "train_pred_loss": train_pred,
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
        The trained joint model after the final epoch.

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

    def on_epoch_end(epoch, train_loss, model):
        """Monitor first, then hand the epoch to the caller's observer."""
        base_on_epoch_end(epoch, train_loss, model)
        if on_epoch is not None:
            on_epoch(epoch, epoch_monitor)

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
    """B1: Two-stage baseline — train imputer only with reconstruction loss.

    The predictor takes no part in the update, but the full training prediction
    objective is evaluated before and after every optimizer step so the
    reconstruction-only trajectory carries the same one-step prediction-loss
    record the TIP branches do.  Its third
    argument repeats ``pred_loss_after_raw_adamw`` because B1 applies the raw
    update unchanged, so ``tip_projection_gain`` is zero on every step by
    construction and the non-increase rate is read off
    ``prediction_step/raw_adamw_pred_loss_delta``.
    """
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
