"""Predictor model registry and configuration converters.

Follows the same pattern as ``src.models.imputators``:
- A ``PREDICTOR_REGISTRY`` mapping model names to ``nn.Module`` classes.
- A ``CONFIG_CONVERTER_REGISTRY`` mapping model names to callables that
  extract constructor kwargs from a ``ConfigParser``.
- A factory function ``predictor(name, config)``.
"""

from __future__ import annotations

from collections.abc import Callable
from configparser import ConfigParser
from typing import Any

import torch.nn as nn

PREDICTOR_REGISTRY: dict[str, str] = {
    "Sundial": ".Sundial.model",
    "TimesFM25": ".TimesFM25.model",
}


def config2param_dict4Sundial(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to SundialPredictor constructor parameters.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Dictionary of SundialPredictor constructor arguments.

    """
    return {
        "model_name": config.get(
            "predictor", "model_name", fallback="thuml/sundial-base-128m"
        ),
        "num_samples": config.getint("predictor", "num_samples", fallback=100),
        "sample_aggregation": config.get(
            "predictor", "sample_aggregation", fallback="mean"
        ),
        "forecast_length": config.getint("dataset", "pred_len"),
        "device": config.get("training", "device"),
    }


def config2param_dict4TimesFM25(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to TimesFM25Predictor constructor parameters."""
    return {
        "model_name": config.get(
            "predictor", "model_name", fallback="google/timesfm-2.5-200m-pytorch"
        ),
        "forecast_length": config.getint("dataset", "pred_len"),
        "max_context": config.getint("predictor", "max_context", fallback=1024),
        "per_core_batch_size": config.getint("training", "batch_size"),
        "normalize_inputs": config.getboolean(
            "predictor", "normalize_inputs", fallback=True
        ),
        "use_continuous_quantile_head": config.getboolean(
            "predictor", "use_continuous_quantile_head", fallback=True
        ),
        "force_flip_invariance": config.getboolean(
            "predictor", "force_flip_invariance", fallback=True
        ),
        "infer_is_positive": config.getboolean(
            "predictor", "infer_is_positive", fallback=True
        ),
        "fix_quantile_crossing": config.getboolean(
            "predictor", "fix_quantile_crossing", fallback=True
        ),
        "device": config.get("training", "device"),
    }


CONFIG_CONVERTER_REGISTRY: dict[str, Callable] = {
    "Sundial": config2param_dict4Sundial,
    "TimesFM25": config2param_dict4TimesFM25,
}


def predictor(predictor_name: str, config: ConfigParser) -> nn.Module:
    """Instantiate a predictor by name using configuration-driven arguments.

    Args:
        predictor_name: Predictor name registered in the predictor registry.
        config: Parsed experiment configuration.

    Returns:
        Instantiated predictor module (inference-only).

    Raises:
        ValueError: If ``predictor_name`` is not registered.

    """
    import importlib

    if predictor_name not in PREDICTOR_REGISTRY:
        available = ", ".join(PREDICTOR_REGISTRY.keys())
        raise ValueError(
            f"Predictor '{predictor_name}' not found. Available predictors: {available}"
        )

    module = importlib.import_module(
        PREDICTOR_REGISTRY[predictor_name], package=__package__
    )
    cls = getattr(module, f"{predictor_name}Predictor")

    param_dict = CONFIG_CONVERTER_REGISTRY[predictor_name](config)

    return cls(**param_dict)
