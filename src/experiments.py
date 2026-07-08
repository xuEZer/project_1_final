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

import hashlib
import logging
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

from src.evaluator import compute_all_losses
from src.models.jointModel import JointModel


class EpochMonitor:
    """Track the best imputer checkpoint and log losses on all splits.

    At every epoch end the monitor computes imputation and prediction MSE
    on the training, validation, and test splits, logs a single concise
    line, and writes scalars to TensorBoard.

    Parameters
    ----------
    joint_model : JointModel
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader or None
        Validation dataloader (may be ``None``).
    test_loader : DataLoader or None
        Test dataloader (may be ``None``).
    exp_type : str
        Experiment type; ``"B1"`` checkpoints on validation imputation
        loss, all others on validation prediction loss.
    logger : logging.Logger
    """

    def __init__(
        self,
        joint_model: JointModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        exp_type: str,
        logger: logging.Logger,
    ) -> None:
        self._model = joint_model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._exp_type = exp_type
        self._logger = logger
        self._best_value = float("inf")
        self._best_epoch = 0
        self._best_state: dict[str, torch.Tensor] | None = None

        # B1 checkpoints on imp_loss; all others on pred_loss.
        self._ckpt_metric = "imp_loss" if exp_type == "B1" else "pred_loss"

    # ------------------------------------------------------------------
    def on_epoch_end(
        self, epoch: int, _train_loss: float, model: JointModel
    ) -> None:
        """Evaluate all splits, log, write TensorBoard, and checkpoint."""
        tb = model.tb_summary_writer

        # Collect losses on every available split ---------------------------
        parts: list[str] = []
        for split_name, loader in [
            ("train", self._train_loader),
            ("val", self._val_loader),
            ("test", self._test_loader),
        ]:
            if loader is None:
                continue
            losses = compute_all_losses(model, loader)
            imp = losses["imp_loss"]
            pred = losses["pred_loss"]
            parts.append(f"{split_name}_imp={imp:.6f} {split_name}_pred={pred:.6f}")
            tb.add_scalar(f"loss/{split_name}_imp", imp, epoch)
            tb.add_scalar(f"loss/{split_name}_pred", pred, epoch)

        self._logger.info("Epoch %03d | %s", epoch, " | ".join(parts))

        # Checkpoint --------------------------------------------------------
        if self._val_loader is not None:
            val_losses = compute_all_losses(model, self._val_loader)
            current = val_losses[self._ckpt_metric]
            if current < self._best_value:
                self._best_value = current
                self._best_epoch = epoch
                self._best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.imputer.state_dict().items()
                }

    # ------------------------------------------------------------------
    def restore_best(self) -> None:
        """Load the best imputer state back into the model."""
        if self._best_state is None:
            return
        self._model.imputer.load_state_dict(self._best_state)
        self._model.to(self._model.device)
        self._logger.info(
            "Restored best val checkpoint: epoch=%03d, %s=%.6f",
            self._best_epoch, self._ckpt_metric, self._best_value,
        )


def run_experiment(
    exp_type: str,
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    test_loader: DataLoader | None = None,
) -> JointModel:
    """Run the configured experiment type.

    Args:
        exp_type: Experiment type identifier (e.g. ``"B2"``).
        joint_model: Initialised joint model (imputer + frozen predictor).
        train_loader: Dictionary with ``"train"`` DataLoader and
            optionally ``"val"``.
        config: Parsed experiment configuration.
        logger: Logger instance.
        test_loader: Optional test-set dataloader for per-epoch logging.

    Returns:
        The trained joint model after the final epoch.

    Raises:
        ValueError: If ``exp_type`` is not recognised.
    """
    scheduler_name = config.get("training", "lr_scheduler", fallback="none")
    min_lr = config.getfloat("training", "min_lr", fallback=0.0)
    num_epoch = config.getint("training", "num_epoch")
    joint_model.configure_lr_scheduler(scheduler_name, num_epoch, min_lr)

    logger.info("Initial imputer state hash: %s", imputer_state_hash(joint_model))

    epoch_monitor = EpochMonitor(
        joint_model,
        train_loader=train_loader.get("train"),
        val_loader=train_loader.get("val"),
        test_loader=test_loader,
        exp_type=exp_type,
        logger=logger,
    )
    on_epoch_end = epoch_monitor.on_epoch_end

    if exp_type == "B1":
        trained_model = _run_b1(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B2":
        trained_model = _run_b2(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B3":
        trained_model = _run_b3(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B4":
        trained_model = _run_b4(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B5":
        trained_model = _run_b5(joint_model, train_loader, config, logger, on_epoch_end)
    elif exp_type == "B6":
        trained_model = _run_b6(joint_model, train_loader, config, logger, on_epoch_end)
    else:
        available = "B1, B2, B3, B4, B5, B6"
        raise ValueError(
            f"Unknown experiment type {exp_type}. Available types: {available}"
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
    num_epoch = config.getint("training", "num_epoch")

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
        on_epoch_end=on_epoch_end,
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
    on_epoch_end=None,
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
            on_epoch_end=on_epoch_end,
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
            on_epoch_end=on_epoch_end,
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
    on_epoch_end=None,
) -> JointModel:
    """B4: Random-direction ablation — random vector replaces ĝ_P."""
    num_epoch = config.getint("training", "num_epoch")

    log_phase(logger, "B4 — Random Direction Ablation", num_epoch)
    joint_model.fit_tip(
        data_loader=train_loader,
        phase_name="b4",
        num_epoch=num_epoch,
        variant="B4",
        on_epoch_end=on_epoch_end,
    )

    return joint_model


# ------------------------------------------------------------------ B5 --


def _run_b5(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
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
        on_epoch_end=on_epoch_end,
        k=k,
        epsilon=epsilon,
    )

    return joint_model


# ------------------------------------------------------------------ B6 --


def _run_b6(
    joint_model: JointModel,
    train_loader: dict[str, DataLoader],
    config: ConfigParser,
    logger: logging.Logger,
    on_epoch_end=None,
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
