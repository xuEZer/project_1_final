"""TimesNet adapter that exposes the project-standard imputer interface.

Paper: https://openreview.net/pdf?id=ju_Uqw384Oq
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from src.models.extern.SAITS.utils import masked_mae_cal
from src.models.extern.TimesNet.TimesNet import Model


class TimesNet(nn.Module):
    """Adapter around TimesNet's ``Model`` with project-compatible IO."""

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
        """Initialize TimesNet backbone and adapter options.

        Args:
            d_feature: Number of input features.
            d_time: Input sequence length.
            d_model: Hidden dimension.
            d_ff: Inner dimension for TimesBlock convolutions.
            top_k: Number of top period lengths from FFT.
            num_kernels: Number of Inception convolution kernels.
            e_layers: Number of TimesBlocks.
            dropout: Dropout rate.
            embed: Embedding type ('fixed' or 'timeF').
            freq: Time frequency for temporal embedding.
            device: Runtime device string.

        """
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
        self.device = device

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run TimesNet imputation and return project-standard output fields.

        Args:
            X_NA: Input with artificial+natural missing values [B, T, D].
            X_NA_mask: Binary mask (1=missing, 0=observed) [B, T, D].
            X_N: Ground-truth complete values [B, T, D] (available during training).
            X_A_mask: Binary mask for natural missing positions [B, T, D].

        Returns:
            Dictionary with ``imp_loss`` and ``imp_data``.

        """
        # Project mask convention: 1=missing, 0=observed.
        # TimesNet convention: mask=1 means observed.
        observed_mask = (1 - X_NA_mask).to(X_NA.dtype)

        dec_out = self.backbone(X_NA, None, None, None, mask=observed_mask)

        # Compute reconstruction loss on raw model output at observed positions
        reconstruction_loss = masked_mae_cal(dec_out, X_NA, observed_mask)

        # Preserve observed positions, replace missing positions with model output
        imputed_data = observed_mask * X_NA + (1 - observed_mask) * dec_out
        loss = reconstruction_loss
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mae_cal(imputed_data, X_N, X_A_mask)

        return {
            "imp_loss": loss,
            "imp_data": imputed_data,
        }
