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
from torch.utils.tensorboard import SummaryWriter

from src.auto_batch_size import find_max_batch_size
from src.data_loader import (
    get_tensor_data_loader,
    load_h5,
    preprocess_dataset,
)
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor
from src.rng import (
    configure_deterministic_backend,
    torch_generator,
    torch_seed_scope,
)

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
    seed: int,
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
        auto_probe_generator = torch_generator(
            seed, "auto_batch_probe_loader_shuffle"
        )
        logger.info(
            "Independent RNG: auto_batch_probe_loader_shuffle_seed=%d",
            auto_probe_generator.initial_seed(),
        )
        sample_loader, _, _ = get_tensor_data_loader(
            dataset,
            batch_size=effective_batch_size,
            train_generator=auto_probe_generator,
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
    seed: int,
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
        seed=seed,
        effective_batch_size=effective_batch_size,
        dataset=dataset,
        joint_model=joint_model,
        loader_kwargs=loader_kwargs,
        logger=logger,
    )
    joint_model.accum_steps = accum_steps
    joint_model.micro_batch_size = micro_batch_size

    train_shuffle_generator = torch_generator(seed, "train_loader_shuffle")
    logger.info(
        "Independent RNG: train_loader_shuffle_seed=%d",
        train_shuffle_generator.initial_seed(),
    )
    train_loader, eval_loader, no_gt = get_tensor_data_loader(
        dataset,
        batch_size=micro_batch_size,
        train_generator=train_shuffle_generator,
        **loader_kwargs,
    )
    return train_loader, eval_loader, no_gt

def configure_deterministic_runtime(seed: int, logger: logging.Logger) -> None:
    """Configure deterministic backends without sharing one global RNG stream."""
    configure_deterministic_backend()
    logger.info(
        "Base random seed set to %d; stochastic processes derive independent RNGs.",
        seed,
    )

def init_joint_model(
    config: ConfigParser, logger: logging.Logger, tb_summary_writer: SummaryWriter
) -> JointModel:
    base_seed = config.getint("experiment", "seed")
    with torch_seed_scope(
        base_seed,
        "imputer_init",
        config.get("imputer", "model"),
    ) as imputer_seed:
        logger.info("Independent RNG: imputer_init_seed=%d", imputer_seed)
        imputer_model = imputator(config.get("imputer", "model"), config)

    with torch_seed_scope(
        base_seed,
        "predictor_init",
        config.get("predictor", "model"),
    ) as predictor_seed:
        logger.info("Independent RNG: predictor_init_seed=%d", predictor_seed)
        predictor_model = predictor(config.get("predictor", "model"), config)

    joint_model = JointModel(
        imputer=imputer_model,
        predictor=predictor_model,
        optimizer_name=config.get("training", "optimizer"),
        lr=config.getfloat("training", "lr"),
        device=config.get("training", "device"),
        tb_summary_writer=tb_summary_writer,
        logger=logger,
    )
    joint_model.configure_random_generators(base_seed)

    device = config.get("training", "device")
    joint_model.to(device)

    return joint_model
