"""Model and runtime construction shared by the expman pipeline."""

from __future__ import annotations

import logging
from configparser import ConfigParser

from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor
from src.rng import configure_deterministic_backend


def parse_column_index(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "all":
        return normalized
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"dataset.column_index must be an integer or 'all', got {value!r}"
        ) from exc


def resolve_missing_seed(config: ConfigParser) -> int:
    """Return the independent seed used to generate missing-data masks."""
    return config.getint("experiment", "missing_seed")


def configure_deterministic_runtime(seed: int, logger: logging.Logger) -> None:
    """Select the deterministic kernels; the RNG stream belongs to expman."""
    configure_deterministic_backend()
    logger.info(
        "Base random seed %d; expman owns the process RNG stream and restores it "
        "at every Stage boundary.",
        seed,
    )


def init_joint_model(config: ConfigParser) -> JointModel:
    """Build the imputer and load the frozen forecaster in an isolated RNG scope.

    The imputer consumes the seeded training stream. Forecaster loading leaves
    that stream unchanged, preserving matched initialization and training draws.
    """
    from src.rng import torch_seed_scope

    base_seed = config.getint("experiment", "seed")
    imputer_model = imputator(config.get("imputer", "model"), config)
    with torch_seed_scope(base_seed, "predictor_load"):
        predictor_model = predictor(config.get("predictor", "model"), config)

    joint_model = JointModel(
        imputer=imputer_model,
        predictor=predictor_model,
        device=config.get("training", "device"),
    )
    joint_model.configure_base_seed(base_seed)

    device = config.get("training", "device")
    joint_model.to(device)

    return joint_model
