"""TimesNet imputer adapter.

Paper: https://openreview.net/pdf?id=ju_Uqw384Oq
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from src.models.extern.SAITS.utils import masked_mse_cal
from src.models.extern.TimesNet.TimesNet import Model


class TimesNet(nn.Module):
    def __init__(
        self,
        d_feature: int,
        d_time: int,
        d_model: int,
        d_ff: int,
        top_k: int,
        num_kernels: int,
        e_layers: int,
        dropout: float,
        embed: str,
        freq: str,
        device: str,
    ):
        super().__init__()
        configs = Namespace(
            task_name="imputation",
            seq_len=d_time,
            label_len=0,
            pred_len=0,
            enc_in=d_feature,
            d_model=d_model,
            embed=embed,
            freq=freq,
            dropout=dropout,
            e_layers=e_layers,
            d_ff=d_ff,
            top_k=top_k,
            num_kernels=num_kernels,
            c_out=d_feature,
            num_class=0,
        )
        self.backbone = Model(configs)

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # X_NA_mask marks missing values as 1; TimesNet's mask marks observed as 1.
        observed_mask = (1 - X_NA_mask).to(X_NA.dtype)

        dec_out = self.backbone(X_NA, None, None, None, mask=observed_mask)
        reconstruction_loss = masked_mse_cal(dec_out, X_NA, observed_mask)
        imputed_data = observed_mask * X_NA + (1 - observed_mask) * dec_out
        loss = reconstruction_loss
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mse_cal(imputed_data, X_N, X_A_mask)

        return {
            "imp_loss": loss,
            "imp_data": imputed_data,
        }
