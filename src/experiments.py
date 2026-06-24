"""Experiment scheduling — phases of imputer training.

Supported experiment types (``exp_type``):

- ``B1``: Two-stage baseline — imputer only, reconstruction loss L_I.
- ``B2``: TIP — zero-order ĝ_P + conditional orthogonal projection.
- ``B3``: Pre-training (2 epochs L_I) then TIP fine-tuning.
- ``B4``: Random-direction ablation (random vector replaces ĝ_P).
- ``B5``: Unconditional-projection ablation (always project).
"""

from __future__ import annotations

import logging
from configparser import ConfigParser

from torch.utils.data import DataLoader

from src.models.jointModel import JointModel


def run_experiment(
    exp_type: str,
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """Run the configured experiment type.

    Args:
        exp_type: Experiment type identifier (e.g. ``"B2"``).
        joint_model: Initialised joint model (imputer + frozen predictor).
        train_loader: Dictionary with ``"train"`` and ``"val"``
            DataLoaders.
        config: Parsed experiment configuration.
        logger: Logger instance.

    Returns:
        The trained joint model (best checkpoint loaded).

    Raises:
        ValueError: If ``exp_type`` is not recognised.
    """
    if exp_type == "B1":
        return _run_b1(joint_model, train_loader, config, logger)
    elif exp_type == "B2":
        return _run_b2(joint_model, train_loader, config, logger)
    elif exp_type == "B3":
        return _run_b3(joint_model, train_loader, config, logger)
    elif exp_type == "B4":
        return _run_b4(joint_model, train_loader, config, logger)
    elif exp_type == "B5":
        return _run_b5(joint_model, train_loader, config, logger)
    else:
        available = "B1, B2, B3, B4, B5"
        raise ValueError(
            f"Unknown experiment type '{exp_type}'. Available types: {available}"
        )


# ------------------------------------------------------------------ B1 --


def _run_b1(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B1: Two-stage baseline — train imputer only with reconstruction loss.

    No task-aware information.  Same as E1 without mandatory warmup.
    """
    max_epoch = config.getint("training", "max_epoch")
    patience = config.getint("training", "patience")

    log_phase(logger, "B1 — Standard Imputer Training", max_epoch)
    joint_model.fit(
        data_loader=train_loader,
        phase_name="b1",
        max_epoch=max_epoch,
        patience=patience,
    )

    return joint_model


# ------------------------------------------------------------------ B2 --


def _run_b2(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B2: Full TIP — zero-order ĝ_P estimation + conditional projection."""
    max_epoch = config.getint("training", "max_epoch")
    patience = config.getint("training", "patience")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)

    log_phase(logger, "B2 — TIP Training", max_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b2",
        max_epoch=max_epoch,
        patience=patience,
        variant="B2",
        k=k,
        epsilon=epsilon,
    )

    return joint_model


# ------------------------------------------------------------------ B3 --


def _run_b3(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B3: Pre-training (L_I only) then TIP fine-tuning."""
    max_epoch = config.getint("training", "max_epoch")
    patience = config.getint("training", "patience")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)
    pretrain_epochs = config.getint("training", "tip_pretrain_epochs", fallback=2)

    # Phase 1: pre-training — no early stopping
    log_phase(logger, "B3 — Pre-training (L_I only)", pretrain_epochs)
    if pretrain_epochs > 0:
        joint_model.fit(
            data_loader=train_loader,
            phase_name="b3_pretrain",
            max_epoch=pretrain_epochs,
            patience=pretrain_epochs,
        )

    # Phase 2: TIP fine-tuning
    log_phase(logger, "B3 — TIP Fine-tuning", max_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b3_tip",
        max_epoch=max_epoch,
        patience=patience,
        variant="B2",
        k=k,
        epsilon=epsilon,
    )

    return joint_model


# ------------------------------------------------------------------ B4 --


def _run_b4(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B4: Random-direction ablation — random vector replaces ĝ_P."""
    max_epoch = config.getint("training", "max_epoch")
    patience = config.getint("training", "patience")

    log_phase(logger, "B4 — Random Direction Ablation", max_epoch)
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b4",
        max_epoch=max_epoch,
        patience=patience,
        variant="B4",
    )

    return joint_model


# ------------------------------------------------------------------ B5 --


def _run_b5(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B5: Unconditional projection — always project, no conflict check."""
    max_epoch = config.getint("training", "max_epoch")
    patience = config.getint("training", "patience")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)

    log_phase(logger, "B5 — Unconditional Projection", max_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b5",
        max_epoch=max_epoch,
        patience=patience,
        variant="B5",
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
