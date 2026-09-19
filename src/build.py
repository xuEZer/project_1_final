"""Model, loader and runtime construction shared by the expman pipeline.

Moved verbatim out of main.py, which held both this construction layer and the
legacy ConfigParser/CLI entry point.  The entry point is gone; these functions
are what src/pipeline.py calls.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.auto_batch_size import find_max_batch_size
from src.data_loader import get_tensor_data_loader
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor
from src.rng import configure_deterministic_backend

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


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
    """Resolve the mask seed while preserving legacy seed behavior."""
    configured = config.get("experiment", "missing_seed", fallback="").strip()
    return int(configured) if configured else config.getint("experiment", "seed")

def resolve_batch_size(
    *,
    config: ConfigParser,
    effective_batch_size: int,
    dataset: dict,
    joint_model: JointModel,
    loader_kwargs: dict,
    logger: logging.Logger,
) -> tuple[int, int]:
    manual_micro_batch = config.getint("loader", "micro_batch")
    if manual_micro_batch > 0:
        micro_batch_size = manual_micro_batch
        accum_steps = math.ceil(effective_batch_size / micro_batch_size)
        logger.info(
            "Using manual micro_batch_size=%d "
            "(target=%d, accum_steps=%d)",
            micro_batch_size,
            effective_batch_size,
            accum_steps,
        )
        return micro_batch_size, accum_steps

    auto_batch = config.has_option("loader", "auto_batch") and config.getboolean(
        "loader", "auto_batch"
    )
    if auto_batch and effective_batch_size > 1:
        sample_loader, _, _ = get_tensor_data_loader(
            dataset,
            batch_size=effective_batch_size,
            **loader_kwargs,
        )
        sample_batch = next(iter(sample_loader["train"]))

        micro_batch_size = find_max_batch_size(
            joint_model,
            sample_batch,
            effective_batch_size,
            config.get("training", "device"),
            logger,
            config,
        )
        accum_steps = math.ceil(effective_batch_size / micro_batch_size)
        logger.info(
            "Effective batch_size: %d | Micro batch_size: %d | "
            "Gradient accumulation steps: %d",
            effective_batch_size,
            micro_batch_size,
            accum_steps,
        )
        return micro_batch_size, accum_steps

    return effective_batch_size, 1

def build_dataloaders(
    config: ConfigParser,
    dataset: dict,
    joint_model: JointModel,
    logger: logging.Logger,
) -> tuple[dict[str, DataLoader], DataLoader, bool]:
    effective_batch_size = config.getint("loader", "batch_size")
    device = config.get("training", "device")
    num_workers = config.getint("loader", "num_workers")
    pin_memory = (
        config.getboolean("loader", "pin_memory")
        and str(device).startswith("cuda")
    )
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}

    micro_batch_size, accum_steps = resolve_batch_size(
        config=config,
        effective_batch_size=effective_batch_size,
        dataset=dataset,
        joint_model=joint_model,
        loader_kwargs=loader_kwargs,
        logger=logger,
    )
    joint_model.accum_steps = accum_steps
    joint_model.micro_batch_size = micro_batch_size

    train_loader, eval_loader, no_gt = get_tensor_data_loader(
        dataset,
        batch_size=micro_batch_size,
        **loader_kwargs,
    )
    return train_loader, eval_loader, no_gt

def configure_deterministic_runtime(seed: int, logger: logging.Logger) -> None:
    """Select the deterministic kernels; the RNG stream belongs to expman."""
    configure_deterministic_backend()
    logger.info(
        "Base random seed %d; expman owns the process RNG stream and restores it "
        "at every Stage boundary.",
        seed,
    )

def init_joint_model(
    config: ConfigParser, logger: logging.Logger, record_metric=None
) -> JointModel:
    base_seed = config.getint("experiment", "seed")
    imputer_model = imputator(config.get("imputer", "model"), config)
    predictor_model = predictor(config.get("predictor", "model"), config)

    joint_model = JointModel(
        imputer=imputer_model,
        predictor=predictor_model,
        optimizer_name=config.get("training", "optimizer"),
        lr=config.getfloat("training", "lr"),
        device=config.get("training", "device"),
        record_metric=record_metric,
        logger=logger,
    )
    joint_model.configure_base_seed(base_seed)

    device = config.get("training", "device")
    joint_model.to(device)

    return joint_model
