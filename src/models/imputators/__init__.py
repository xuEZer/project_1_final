"""Imputer model registry and configuration converters."""

from collections.abc import Callable
from configparser import ConfigParser
from typing import Any

import torch.nn as nn

from .BRITS import BRITS
from .DeepMVI import DeepMVI
from .SAITS import SAITS
from .TimesNet import TimesNet

IMPUTATOR_REGISTRY: dict[str, type] = {
    "SAITS": SAITS,
    "BRITS": BRITS,
    "DeepMVI": DeepMVI,
    "TimesNet": TimesNet,
}


def config2param_dict4SAITS(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to SAITS constructor parameters.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Dictionary of SAITS constructor arguments.

    """
    param_dict = {
        "n_groups": int(config["imputer"]["n_groups"]),
        "n_group_inner_layers": int(config["imputer"]["n_group_inner_layers"]),
        "d_time": int(config["data"]["lookback"]),
        "d_feature": int(config["data"]["d_feature"]),
        "d_model": int(config["imputer"]["d_model"]),
        "d_inner": int(config["imputer"]["d_inner"]),
        "n_head": int(config["imputer"]["n_head"]),
        "d_k": int(config["imputer"]["d_k"]),
        "d_v": int(config["imputer"]["d_v"]),
        "dropout": float(config["imputer"]["dropout"]),
        "param_sharing_strategy": config["imputer"]["param_sharing_strategy"],
        "diagonal_attention_mask": config["imputer"].getboolean(
            "diagonal_attention_mask"
        ),
        "device": config["training"]["device"],
    }
    return param_dict


def config2param_dict4BRITS(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to BRITS constructor parameters.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Dictionary of BRITS constructor arguments.

    """
    param_dict = {
        "d_feature": int(config["data"]["d_feature"]),
        "d_model": int(config["imputer"]["d_model"]),
        "device": config["training"]["device"],
    }
    return param_dict


def config2param_dict4DeepMVI(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to DeepMVI constructor parameters.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Dictionary of DeepMVI constructor arguments.

    """
    param_dict = {
        "d_feature": int(config["data"]["d_feature"]),
        "d_time": int(config["data"]["lookback"]),
        "device": config["training"]["device"],
        "kernel_size": int(config["imputer"].get("kernel_size", 10)),
        "block_size": int(config["imputer"].get("block_size", 5)),
        "n_head": int(config["imputer"].get("n_head", 2)),
        "use_context": config["imputer"].getboolean("use_context", fallback=True),
        "use_local": config["imputer"].getboolean("use_local", fallback=True),
    }
    return param_dict


def config2param_dict4TimesNet(config: ConfigParser) -> dict[str, Any]:
    """Convert configuration values to TimesNet constructor parameters.

    Args:
        config: Parsed experiment configuration.

    Returns:
        Dictionary of TimesNet constructor arguments.

    """
    param_dict = {
        "d_feature": int(config["data"]["d_feature"]),
        "d_time": int(config["data"]["lookback"]),
        "d_model": int(config["imputer"]["d_model"]),
        "d_ff": int(config["imputer"]["d_ff"]),
        "top_k": int(config["imputer"]["top_k"]),
        "num_kernels": int(config["imputer"]["num_kernels"]),
        "e_layers": int(config["imputer"]["e_layers"]),
        "dropout": float(config["imputer"]["dropout"]),
        "embed": config["imputer"]["embed"],
        "freq": config["imputer"]["freq"],
        "device": config["training"]["device"],
    }
    return param_dict


CONFIG_CONVERTER_REGISTRY: dict[str, Callable] = {
    "SAITS": config2param_dict4SAITS,
    "BRITS": config2param_dict4BRITS,
    "DeepMVI": config2param_dict4DeepMVI,
    "TimesNet": config2param_dict4TimesNet,
}


def imputator(imputator_name: str, config: ConfigParser) -> nn.Module:
    """Instantiate an imputer by name using configuration-driven arguments.

    Args:
        imputator_name: Imputer name registered in the imputer registry.
        config: Parsed experiment configuration.

    Returns:
        Instantiated imputer module.

    """
    if imputator_name not in IMPUTATOR_REGISTRY:
        available = ", ".join(IMPUTATOR_REGISTRY.keys())
        raise ValueError(
            f"Imputator '{imputator_name}' not found. Available imputators: {available}"
        )

    param_dict = CONFIG_CONVERTER_REGISTRY[imputator_name](config)

    return IMPUTATOR_REGISTRY[imputator_name](**param_dict)
