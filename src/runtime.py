"""Configuration conversion and deterministic dataset preparation."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from configparser import ConfigParser, ExtendedInterpolation

import torch

_CPU_THREADS = max(1, int(os.environ.get("PILOT_CPU_THREADS", "2")))
torch.set_num_threads(_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _scalar(value):
    return "" if value is None else str(value)


def _plain(value):
    """Convert expman's immutable configuration containers to Python values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_plain(item) for item in value]
    return value


def build_config(run, **runtime):
    """Build model configuration with dataset dimensions and the worker device."""
    values = _plain(run)
    values.update(values.pop("models"))
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.add_section("experiment")
    for key, value in values.items():
        section, fields = (
            (key, value) if isinstance(value, Mapping) else ("experiment", {key: value})
        )
        if not parser.has_section(section):
            parser.add_section(section)
        for field, item in fields.items():
            parser.set(section, field, _scalar(item))
    parser.set("training", "device", "cuda:0")
    parser.set("data", "dataset", _scalar(values["dataset"]["name"]))
    for section, fields in runtime.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for field, item in fields.items():
            parser.set(section, field, _scalar(item))
    return parser


def prepare_dataset(run: dict) -> dict:
    """Build the windowed tensors for one run configuration.

    Shared by :class:`PrepareData` and :class:`Evaluate`.  Both the raw-series
    lookup and the mask streams are deterministic in the configuration -- the
    primary synthetic and artificial masks come from RNG streams derived from
    ``missing_seed`` -- so calling this twice for the same run returns identical
    tensors.
    """
    from src.build import parse_column_index, resolve_missing_seed
    from src.data_loader import load_raw_series, preprocess_dataset

    cfg = build_config(run)
    dataset = run["dataset"]
    series = load_raw_series(
        source=dataset["source"],
        repo_name=dataset["repo_name"],
        dataset_name=dataset["dataset_name"],
        frequency=dataset["frequency"],
        item_id=dataset["item_id"],
        column_index=parse_column_index(_scalar(dataset["column_index"])),
        cache_dir=cfg.get("paths", "cache_dir"),
    )
    return preprocess_dataset(
        series=series,
        missing_rate=cfg.getfloat("data", "missing_rate"),
        artificial_missing_rate=cfg.getfloat("data", "artificial_missing_rate"),
        block_size=cfg.getint("data", "block_size"),
        lookback_len=cfg.getint("data", "lookback"),
        pred_len=cfg.getint("data", "pred_len"),
        val_ratio=cfg.getfloat("data", "val_ratio"),
        test_ratio=cfg.getfloat("data", "test_ratio"),
        random_seed=resolve_missing_seed(cfg),
        logger=logging.getLogger("expman.pipeline"),
    )


def runtime_from_dataset(dataset):
    """Return the feature count and context length of the prepared data."""
    return {
        "d_feature": str(dataset["train"]["X_N"].shape[2]),
        "lookback": str(dataset["train"]["X_N"].shape[1]),
    }


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value
