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
from .Timer.model import TimerPredictor
from .Toto2.model import Toto2Predictor

PREDICTOR_REGISTRY: dict[str, type] = {
    "Sundial": SundialPredictor,
    "Chronos": ChronosPredictor,
    "Timer": TimerPredictor,
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


def _config2timer(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "forecast_length": config.getint("data", "pred_len"),
        "normalize_inputs": config.getboolean("predictor", "normalize_inputs"),
        "device": config.get("training", "device"),
        "inference_device": config.get(
            "predictor", "inference_device", fallback=None,
        ),
    }


def _config2toto2(config: ConfigParser) -> dict[str, Any]:
    return {
        "model_name": config.get("predictor", "model_name"),
        "forecast_length": config.getint("data", "pred_len"),
        "device": config.get("training", "device"),
    }


CONFIG_CONVERTER_REGISTRY: dict[str, Callable] = {
    "Sundial": _config2sundial,
    "Chronos": _config2chronos,
    "Timer": _config2timer,
    "Toto2": _config2toto2,
}


def predictor(predictor_name: str, config: ConfigParser) -> nn.Module:
    """Instantiate a predictor by name using configuration-driven arguments.

    Args:
        predictor_name: ``"Sundial"``, ``"Chronos"``, ``"Timer"``, or ``"Toto2"``.
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
