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
        "n_groups": int(config["imputator"]["n_groups"]),
        "n_group_inner_layers": int(config["imputator"]["n_group_inner_layers"]),
        "d_time": int(config["dataset"]["lookback_len"]),
        "d_feature": int(config["dataset"]["d_feature"]),
        "d_model": int(config["imputator"]["d_model"]),
        "d_inner": int(config["imputator"]["d_inner"]),
        "n_head": int(config["imputator"]["n_head"]),
        "d_k": int(config["imputator"]["d_k"]),
        "d_v": int(config["imputator"]["d_v"]),
        "dropout": float(config["imputator"]["dropout"]),
        "param_sharing_strategy": config["imputator"]["param_sharing_strategy"],
        "diagonal_attention_mask": config["imputator"].getboolean(
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
        "d_feature": int(config["dataset"]["d_feature"]),
        "d_model": int(config["imputator"]["d_model"]),
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
        "d_feature": int(config["dataset"]["d_feature"]),
        "d_time": int(config["dataset"]["lookback_len"]),
        "device": config["training"]["device"],
        "kernel_size": int(config["imputator"].get("kernel_size", 10)),
        "block_size": int(config["imputator"].get("block_size", 5)),
        "n_head": int(config["imputator"].get("n_head", 2)),
        "use_context": config["imputator"].getboolean("use_context", fallback=True),
        "use_local": config["imputator"].getboolean("use_local", fallback=True),
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
        "d_feature": int(config["dataset"]["d_feature"]),
        "d_time": int(config["dataset"]["lookback_len"]),
        "d_model": int(config["imputator"]["d_model"]),
        "d_ff": int(config["imputator"]["d_ff"]),
        "top_k": int(config["imputator"]["top_k"]),
        "num_kernels": int(config["imputator"]["num_kernels"]),
        "e_layers": int(config["imputator"]["e_layers"]),
        "dropout": float(config["imputator"]["dropout"]),
        "embed": config["imputator"]["embed"],
        "freq": config["imputator"]["freq"],
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
