"""Predictor model registry and configuration converters."""

from __future__ import annotations

import os
from collections.abc import Callable
from configparser import ConfigParser
from typing import Any

import torch.nn as nn

# huggingface_hub caches HF_ENDPOINT at import time, so the mirror must be
# set before the predictor imports below; setdefault honors an explicit user
# setting.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .Chronos.model import ChronosPredictor
from .Sundial.model import SundialPredictor
from .TimesFM3.model import TimesFM3Predictor
from .Toto2.model import Toto2Predictor

PREDICTOR_REGISTRY: dict[str, type] = {
    "Sundial": SundialPredictor,
    "Chronos": ChronosPredictor,
    "TimesFM3": TimesFM3Predictor,
    "Toto2": Toto2Predictor,
}


def _config2sundial(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "num_samples": config.getint("predictor", "num_samples"),
        "sample_aggregation": config.get("predictor", "sample_aggregation"),
        "forecast_length": config.getint("data", "pred_len"),
        "device": config.get("training", "device"),
    }


def _config2chronos(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "forecast_length": config.getint("data", "pred_len"),
        "num_samples": config.getint("predictor", "num_samples"),
        "device": config.get("training", "device"),
    }


def _config2toto2(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "forecast_length": config.getint("data", "pred_len"),
        "device": config.get("training", "device"),
    }


def _config2timesfm3(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "forecast_length": config.getint("data", "pred_len"),
        "device": config.get("training", "device"),
        "per_core_batch_size": config.getint("predictor", "per_core_batch_size"),
    }


CONFIG_CONVERTER_REGISTRY: dict[str, Callable] = {
    "Sundial": _config2sundial,
    "Chronos": _config2chronos,
    "TimesFM3": _config2timesfm3,
    "Toto2": _config2toto2,
}


def predictor(predictor_name: str, config: ConfigParser) -> nn.Module:
    """Instantiate a predictor by name using configuration-driven arguments.

    Args:
        predictor_name: Registered predictor name, such as ``"TimesFM3"``.
        config: Parsed experiment configuration.

    Returns:
        Instantiated predictor module (inference-only).

    Raises:
        ValueError: If *predictor_name* is not registered.

    """
    if predictor_name not in PREDICTOR_REGISTRY:
        available = ", ".join(PREDICTOR_REGISTRY.keys())
        raise ValueError(
            f"Predictor '{predictor_name}' not found. "
            f"Available predictors: {available}"
        )

    param_dict = CONFIG_CONVERTER_REGISTRY[predictor_name](config)
    return PREDICTOR_REGISTRY[predictor_name](**param_dict)
