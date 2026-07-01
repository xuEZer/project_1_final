"""Experiment scheduling — phases of imputer training.

Supported experiment types (``exp_type``):

- ``B1``: Two-stage baseline — imputer only, reconstruction loss L_I.
- ``B2``: TIP — zero-order ĝ_P + conditional orthogonal projection.
- ``B3``: Pre-training with L_I, then TIP fine-tuning within the total
  epoch budget.
- ``B4``: Random-direction ablation (random vector replaces ĝ_P).
- ``B5``: Unconditional-projection ablation (always project).
- ``B6``: Direct prediction-direction update (use ĝ_P as the optimizer
  gradient).
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
        The trained joint model after the final epoch.

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
    elif exp_type == "B6":
        return _run_b6(joint_model, train_loader, config, logger)
    else:
        available = "B1, B2, B3, B4, B5, B6"
        raise ValueError(
            f"Unknown experiment type '{exp_type}'. Available types: {available}"
        )


def tip_projection_kwargs(config: ConfigParser) -> dict[str, bool | float]:
    """Read TIP projection controls from the training config."""
    return {
        "preserve_grad_norm": config.getboolean(
            "training", "tip_preserve_grad_norm", fallback=True
        ),
        "norm_scale_cap": config.getfloat(
            "training", "tip_norm_scale_cap", fallback=0.0
        ),
    }


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
    num_epoch = config.getint("training", "num_epoch")

    log_phase(logger, "B1 — Standard Imputer Training", num_epoch)
    joint_model.fit(
        data_loader=train_loader,
        phase_name="b1",
        num_epoch=num_epoch,
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
    num_epoch = config.getint("training", "num_epoch")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)

    log_phase(logger, "B2 — TIP Training", num_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b2",
        num_epoch=num_epoch,
        variant="B2",
        k=k,
        epsilon=epsilon,
        **tip_projection_kwargs(config),
    )

    return joint_model


# ------------------------------------------------------------------ B3 --


def _run_b3(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B3: Pre-training (L_I only) then TIP fine-tuning within num_epoch."""
    num_epoch = config.getint("training", "num_epoch")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)
    pretrain_epochs = config.getint("training", "tip_pretrain_epochs", fallback=2)
    pretrain_epochs = max(0, pretrain_epochs)
    if pretrain_epochs > num_epoch:
        logger.warning(
            "B3 tip_pretrain_epochs=%d exceeds num_epoch=%d; using %d "
            "pre-training epochs and 0 TIP epochs.",
            pretrain_epochs,
            num_epoch,
            num_epoch,
        )
        pretrain_epochs = num_epoch
    tip_epochs = num_epoch - pretrain_epochs

    logger.info(
        "B3 total epoch budget: %d = %d pre-training + %d TIP fine-tuning",
        num_epoch,
        pretrain_epochs,
        tip_epochs,
    )

    # Phase 1: fixed-epoch pre-training
    log_phase(logger, "B3 — Pre-training (L_I only)", pretrain_epochs)
    if pretrain_epochs > 0:
        joint_model.fit(
            data_loader=train_loader,
            phase_name="b3_pretrain",
            num_epoch=pretrain_epochs,
        )

    # Phase 2: TIP fine-tuning
    log_phase(logger, "B3 — TIP Fine-tuning", tip_epochs)
    if tip_epochs > 0:
        logger.info(f"  k={k}, epsilon={epsilon}")
        joint_model.fit_tip(
            data_loader=train_loader,
            phase_name="b3_tip",
            num_epoch=tip_epochs,
            variant="B2",
            k=k,
            epsilon=epsilon,
            **tip_projection_kwargs(config),
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
    num_epoch = config.getint("training", "num_epoch")

    log_phase(logger, "B4 — Random Direction Ablation", num_epoch)
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b4",
        num_epoch=num_epoch,
        variant="B4",
        **tip_projection_kwargs(config),
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
    num_epoch = config.getint("training", "num_epoch")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)

    log_phase(logger, "B5 — Unconditional Projection", num_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b5",
        num_epoch=num_epoch,
        variant="B5",
        k=k,
        epsilon=epsilon,
        **tip_projection_kwargs(config),
    )

    return joint_model


# ------------------------------------------------------------------ B6 --


def _run_b6(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
) -> JointModel:
    """B6: Directly update the imputer with the estimated prediction gradient."""
    num_epoch = config.getint("training", "num_epoch")
    k = config.getint("training", "tip_k", fallback=20)
    epsilon = config.getfloat("training", "tip_epsilon", fallback=1e-4)

    log_phase(logger, "B6 - Direct Prediction Gradient", num_epoch)
    logger.info(f"  k={k}, epsilon={epsilon}")
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b6",
        num_epoch=num_epoch,
        variant="B6",
        k=k,
        epsilon=epsilon,
    )

    return joint_model


# ------------------------------------------------------------------ util --


def log_phase(logger: logging.Logger, phase: str, epochs: int) -> None:
    """Log a phase boundary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Phase: {phase} ({epochs} fixed epochs)")
    logger.info("=" * 60)
