"""TSLANet imputer adapter.

Backbone adapted from TSLANet: Rethinking Transformers for Time Series
Representation Learning (ICML 2024), following the public PyPOTS imputation
integration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.extern.SAITS.utils import masked_mse_cal


def _drop_path(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0.0 or not training:
        return x
    keep_probability = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_probability).div_(keep_probability)
    return x * random_tensor


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.probability, self.training)


class _InteractiveConvolutionBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.conv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        branch_1 = self.conv1(x)
        branch_2 = self.conv2(x)
        mixed = branch_1 * self.dropout(self.activation(branch_2))
        mixed = mixed + branch_2 * self.dropout(self.activation(branch_1))
        return self.conv3(mixed).transpose(1, 2)


class _AdaptiveSpectralBlock(nn.Module):
    def __init__(self, dim: int, adaptive_filter: bool) -> None:
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.threshold = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        weighted = spectrum * torch.view_as_complex(self.complex_weight)

        if self.adaptive_filter:
            energy = spectrum.abs().square().sum(dim=-1)
            median_energy = energy.flatten(1).median(dim=1, keepdim=True).values
            mask = ((energy / (median_energy + 1e-6) > self.threshold).float() - self.threshold).detach()
            mask = mask + self.threshold
            weighted = weighted + spectrum * mask.unsqueeze(-1) * torch.view_as_complex(self.complex_weight_high)

        return torch.fft.irfft(weighted, n=x.size(1), dim=1, norm="ortho").to(input_dtype)


class _TSLABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        adaptive_filter: bool,
        apply_icb: bool,
        apply_asb: bool,
    ) -> None:
        super().__init__()
        self.apply_icb = apply_icb
        self.apply_asb = apply_asb
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.asb = _AdaptiveSpectralBlock(dim, adaptive_filter)
        self.icb = _InteractiveConvolutionBlock(dim, mlp_ratio, dropout)
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_icb and self.apply_asb:
            return x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        if self.apply_icb:
            return x + self.drop_path(self.icb(self.norm2(x)))
        if self.apply_asb:
            return x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    def __init__(
        self,
        d_feature: int,
        d_time: int,
        n_layers: int,
        patch_size: int,
        d_embedding: int,
        dropout: float,
        adaptive_filter: bool,
        apply_icb: bool,
        apply_asb: bool,
        device: str,
    ) -> None:
        super().__init__()
        if patch_size < 2 or patch_size > d_time:
            raise ValueError("patch_size must be in [2, d_time]")
        self.patch_size = patch_size
        self.stride = patch_size // 2
        self.n_patches = (d_time - patch_size) // self.stride + 1
        self.input_layer = nn.Linear(patch_size, d_embedding)
        self.blocks = nn.ModuleList(
            [
                _TSLABlock(
                    d_embedding,
                    mlp_ratio=3.0,
                    dropout=dropout,
                    drop_path=dropout * index / max(n_layers - 1, 1),
                    adaptive_filter=adaptive_filter,
                    apply_icb=apply_icb,
                    apply_asb=apply_asb,
                )
                for index in range(n_layers)
            ]
        )
        self.output_layer = nn.Linear(d_embedding * self.n_patches, d_time)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_features = x.shape
        patches = x.transpose(1, 2).unfold(-1, self.patch_size, self.stride)
        encoded = self.input_layer(patches.reshape(batch_size * n_features, self.n_patches, self.patch_size))
        for block in self.blocks:
            encoded = block(encoded)
        reconstructed = self.output_layer(encoded.reshape(batch_size * n_features, -1))
        return reconstructed.reshape(batch_size, n_features, -1).transpose(1, 2)

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        observed_mask = (1.0 - X_NA_mask).to(X_NA.dtype)
        reconstruction = self._reconstruct(X_NA)
        imputed_data = observed_mask * X_NA + (1.0 - observed_mask) * reconstruction

        loss = masked_mse_cal(reconstruction, X_NA, observed_mask)
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mse_cal(imputed_data, X_N, X_A_mask)
        return {"imp_loss": loss, "imp_data": imputed_data}
