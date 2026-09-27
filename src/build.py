"""Model, loader and runtime construction shared by the expman pipeline.

Moved verbatim out of main.py, which held both this construction layer and the
legacy ConfigParser/CLI entry point.  The entry point is gone; these functions
are what src/pipeline.py calls.
"""

from __future__ import annotations

import logging
import math
import re
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader

from src.data_loader import get_tensor_data_loader
from src.models.imputators import imputator
from src.models.jointModel import JointModel
from src.models.predictors import predictor
from src.rng import configure_deterministic_backend, derive_seed

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

    # B1 and B2 must see the same shuffled batches.  Keep the loader RNG
    # independent of the ambient Stage stream and of any extra TIP forwards;
    # the namespace is deliberately fixed and contains no branch/stage label.
    shuffle_seed = derive_seed(
        config.getint("experiment", "seed"), "train_loader_shuffle"
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(shuffle_seed)
    logger.info("Training DataLoader shuffle seed: %d", shuffle_seed)

    train_loader, eval_loader, no_gt = get_tensor_data_loader(
        dataset,
        batch_size=micro_batch_size,
        train_generator=train_generator,
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
    """Build the imputer and the frozen predictor for one run.

    The predictor is constructed inside a seed scope because the HuggingFace
    loaders build their modules from the global Torch RNG: without the scope the
    stream that training draws from -- and therefore the training DataLoader's
    shuffle order -- would depend on which predictor is loaded.  The imputer is
    built first and outside the scope, so its initialisation keeps drawing from
    the Stage's own stream and is identical for every run of a given seed.
    """
    from src.rng import torch_seed_scope

    base_seed = config.getint("experiment", "seed")
    imputer_model = imputator(config.get("imputer", "model"), config)
    with torch_seed_scope(base_seed, "predictor_load"):
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
