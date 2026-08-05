"""FITS imputer adapter.

Backbone from the official FITS implementation (ICLR 2024) via its PyPOTS
integration: "FITS: Modeling Time Series with 10k Parameters".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.extern.SAITS.utils import masked_mse_cal


class _FITSBackbone(nn.Module):
    def __init__(self, n_steps: int, n_features: int, cut_freq: int, individual: bool):
        super().__init__()
        if not 1 <= cut_freq <= n_steps // 2 + 1:
            raise ValueError("cut_freq must be in [1, n_steps // 2 + 1]")

        self.n_steps = n_steps
        self.cut_freq = cut_freq
        self.individual = individual

        if individual:
            self.real_upsampler = nn.ModuleList(
                [nn.Linear(cut_freq, cut_freq) for _ in range(n_features)]
            )
            self.imag_upsampler = nn.ModuleList(
                [nn.Linear(cut_freq, cut_freq) for _ in range(n_features)]
            )
        else:
            self.real_upsampler = nn.Linear(cut_freq, cut_freq)
            self.imag_upsampler = nn.Linear(cut_freq, cut_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        low_spectrum = spectrum[:, : self.cut_freq, :]

        if self.individual:
            real = torch.zeros_like(low_spectrum.real)
            imag = torch.zeros_like(low_spectrum.imag)
            for feature_index, (real_layer, imag_layer) in enumerate(
                zip(self.real_upsampler, self.imag_upsampler, strict=True)
            ):
                real[:, :, feature_index] = real_layer(
                    low_spectrum.real[:, :, feature_index]
                )
                imag[:, :, feature_index] = imag_layer(
                    low_spectrum.imag[:, :, feature_index]
                )
            projected = torch.complex(real, imag)
        else:
            real = self.real_upsampler(
                low_spectrum.real.permute(0, 2, 1)
            ).permute(0, 2, 1)
            imag = self.imag_upsampler(
                low_spectrum.imag.permute(0, 2, 1)
            ).permute(0, 2, 1)
            projected = torch.complex(real, imag)

        padded = torch.zeros_like(spectrum)
        padded[:, : self.cut_freq, :] = projected
        return torch.fft.irfft(padded, n=self.n_steps, dim=1)


class FITS(nn.Module):
    def __init__(
        self,
        d_feature: int,
        d_time: int,
        cut_freq: int,
        individual: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(d_feature * 2, d_feature)
        self.backbone = _FITSBackbone(d_time, d_feature, cut_freq, individual)
        self.output_projection = nn.Linear(d_feature, d_feature)

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        observed_mask = (1.0 - X_NA_mask).to(X_NA.dtype)
        embedded = self.embedding(torch.cat([X_NA, observed_mask], dim=-1))
        reconstruction = self.output_projection(self.backbone(embedded))
        imputed_data = observed_mask * X_NA + (1.0 - observed_mask) * reconstruction

        loss = masked_mse_cal(reconstruction, X_NA, observed_mask)
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mse_cal(imputed_data, X_N, X_A_mask)

        return {"imp_loss": loss, "imp_data": imputed_data}
