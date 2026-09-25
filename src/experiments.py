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
    elif exp_type == "RQ3":
        trained_model = _run_rq3(
            joint_model, train_loader, config, logger, on_epoch_end
        )
    elif exp_type == "RQ4":
        trained_model = _run_rq4(
            joint_model, train_loader, config, logger, on_epoch_end
        )
    elif exp_type in {"B2", "B3", "B4", "B5", "B6", "B7"}:
        trained_model = _run_b2(
            joint_model, train_loader, config, logger, on_epoch_end,
            exp_type=exp_type,
        )
    else:
        raise ValueError(
            f"Unknown experiment type {exp_type}. "
            "Available types: B1, B2, B3, B4, B5, B6, B7, RQ3, RQ4"
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
    """Run TIP or a matched subspace/correction ablation."""
    num_epoch = config.getint("training", "max_epochs")
    subspace_rank = config.getint("tip", "subspace_rank")
    epsilon = config.getfloat("tip", "epsilon")
    # ``confidence_delta`` is the legacy shared threshold.  Keep it as a
    # compatibility default, but allow fully decoupled configurations to
    # specify only ``gate_delta`` and ``cone_delta``.
    confidence_delta = config.getfloat(
        "tip", "confidence_delta", fallback=0.05,
    )
    gate_delta = config.getfloat(
        "tip", "gate_delta", fallback=confidence_delta,
    )
    cone_delta = config.getfloat(
        "tip", "cone_delta", fallback=confidence_delta,
    )
    cone_radius = config.get("tip", "cone_radius", fallback="hotelling")
    finite_difference_third_derivative = config.getfloat(
        "tip", "finite_difference_third_derivative", fallback=0.0,
    )
    # Fairness control: give B2 the same displacement norm as B1 so the two
    # arms differ only in the direction of the step.
    preserve_norm = config.getboolean("tip", "preserve_norm", fallback=False)
    # Let reconstruction train the imputer alone for this many epochs before the
    # forecast mechanism is engaged.
    warmup_epochs = config.getint("tip", "warmup_epochs", fallback=0)

    robust_gate_mode = config.get("tip", "gate_mode", fallback="ucb")
    # B6: the full mechanism without the warm-up (engaged from epoch 0).
    # B7: same gate, but the fired correction is replaced by the estimated
    # forecast-gradient direction itself (norm-preserved), the direct
    # "descend Q" control that theory predicts to be weaker.
    variants = {
        "B2": ("history", "robust", robust_gate_mode, {}),
        "B3": ("random", "robust", robust_gate_mode, {}),
        "B4": ("history", "point", robust_gate_mode, {}),
        "B5": ("history", "robust", "mean", {}),
        "B6": ("history", "robust", robust_gate_mode, {"warmup_epochs": 0}),
        "B7": ("history", "gradient", robust_gate_mode, {}),
    }
    anchor_delta = config.getfloat("tip", "anchor_delta", fallback=0.0)
    accept_only_if_improves = config.getboolean(
        "tip", "accept_only_if_improves", fallback=False,
    )
    prediction_probe_batches = config.getint(
        "tip", "prediction_probe_batches", fallback=1,
    )
    basis_mode, correction_mode, gate_mode, overrides = variants[exp_type]
    log_phase(
        logger,
        f"{exp_type} — TIP ({basis_mode} subspace, {correction_mode} "
        f"correction, {gate_mode} gate, overrides={overrides or 'none'})",
        num_epoch,
    )
    logger.info(
        "  subspace_rank=%d, epsilon=%g, gate_delta=%g, cone_delta=%g, "
        "cone_radius=%s, fd_third_derivative=%g, preserve_norm=%s, "
        "warmup_epochs=%d,"
        "basis_mode=%s, correction_mode=%s, gate_mode=%s",
        subspace_rank, epsilon, gate_delta, cone_delta, cone_radius,
        finite_difference_third_derivative, preserve_norm, warmup_epochs,
        basis_mode, correction_mode, gate_mode,
    )
    tip_kwargs = dict(
        warmup_epochs=warmup_epochs,
    )
    tip_kwargs.update(overrides)
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name=exp_type.lower(),
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        subspace_rank=subspace_rank,
        epsilon=epsilon,
        confidence_delta=confidence_delta,
        gate_delta=gate_delta,
        cone_delta=cone_delta,
        cone_radius=cone_radius,
        finite_difference_third_derivative=finite_difference_third_derivative,
        preserve_norm=preserve_norm,
        basis_mode=basis_mode,
        correction_mode=correction_mode,
        gate_mode=gate_mode,
        anchor_delta=anchor_delta,
        accept_only_if_improves=accept_only_if_improves,
        **tip_kwargs,
        prediction_probe_batches=prediction_probe_batches,
        initial_history=initial_history,
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


def _run_rq4(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
) -> JointModel:
    """Train one independently realised RQ4 sensitivity configuration."""
    num_epoch = config.getint("training", "max_epochs")
    variant = config.get("experiment", "rq4_variant")

    if variant == "B1":
        return _run_b1(
            joint_model, train_loader, config, logger, on_epoch_end
        )

    rank = config.getint("tip", "subspace_rank")
    epsilon = config.getfloat("tip", "epsilon")
    confidence_delta = config.getfloat("tip", "confidence_delta")
    if variant == "base":
        pass
    elif variant.startswith("rank="):
        rank = int(variant.split("=", 1)[1])
    elif variant.startswith("epsilon="):
        epsilon = float(variant.split("=", 1)[1])
    elif variant.startswith("gate_delta="):
        confidence_delta = float(variant.split("=", 1)[1])
    else:
        raise ValueError(f"Unknown RQ4 variant: {variant}")

    if rank < 2:
        raise ValueError("RQ4 rank must be at least 2")
    if epsilon <= 0:
        raise ValueError("RQ4 epsilon must be positive")
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("RQ4 gate delta must lie in (0, 1)")

    log_phase(logger, f"RQ4 — independent TIP training ({variant})", num_epoch)
    logger.info(
        "  variant=%s, subspace_rank=%d, epsilon=%g, confidence_delta=%g",
        variant, rank, epsilon, confidence_delta,
    )
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="rq4",
        num_epoch=num_epoch,
        on_epoch_end=on_epoch_end,
        subspace_rank=rank,
        epsilon=epsilon,
        confidence_delta=confidence_delta,
        basis_mode="history",
        correction_mode="robust",
        gate_mode="ucb",
    )
    return joint_model


def log_phase(logger: logging.Logger, phase: str, epochs: int) -> None:
    """Log a phase boundary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase: {phase} ({epochs} max epochs)")
    logger.info("=" * 60)
