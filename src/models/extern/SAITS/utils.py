"""Utility functions are stored here.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu.
SAITS: Self-Attention-based Imputation for Time Series.
Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

import torch


def masked_mae_cal(
    inputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate Mean Absolute Error for masked values.

    Args:
        inputs: Predicted values
        target: Ground truth values
        mask: Binary mask (1 for valid, 0 for invalid)

    Returns:
        Mean absolute error over masked region

    """
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_mse_cal(
    inputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate Mean Square Error for masked values.

    Args:
        inputs: Predicted values
        target: Ground truth values
        mask: Binary mask (1 for valid, 0 for invalid)

    Returns:
        Mean square error over masked region

    """
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(
    inputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate Root Mean Square Error for masked values.

    Args:
        inputs: Predicted values
        target: Ground truth values
        mask: Binary mask (1 for valid, 0 for invalid)

    Returns:
        Root mean square error over masked region

    """
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(
    inputs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate Mean Relative Error for masked values.

    Args:
        inputs: Predicted values
        target: Ground truth values
        mask: Binary mask (1 for valid, 0 for invalid)

    Returns:
        Mean relative error over masked region

    """
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )
