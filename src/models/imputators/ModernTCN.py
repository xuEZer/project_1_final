"""ModernTCN imputer adapter.

Imputation backbone from ModernTCN: A Modern Pure Convolution Structure for
General Time Series Analysis (ICLR 2024), adapted to the project-standard
imputer interface.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.extern.SAITS.utils import masked_mse_cal


def _conv_bn(channels: int, kernel_size: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm1d(channels),
    )


class _ReparamLargeKernelConv(nn.Module):
    def __init__(self, channels: int, large_size: int, small_size: int | None) -> None:
        super().__init__()
        self.large_branch = _conv_bn(channels, large_size, channels)
        self.small_branch = _conv_bn(channels, small_size, channels) if small_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.large_branch(x)
        if self.small_branch is not None:
            output = output + self.small_branch(x)
        return output


class _ModernTCNBlock(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        ffn_ratio: int,
        large_size: int,
        small_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = n_features * d_model
        d_ff = d_model * ffn_ratio
        self.depthwise = _ReparamLargeKernelConv(channels, large_size, small_size)
        self.norm = nn.BatchNorm1d(d_model)
        self.feature_mixing = nn.Sequential(
            nn.Conv1d(channels, n_features * d_ff, 1, groups=n_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_features * d_ff, channels, 1, groups=n_features),
            nn.Dropout(dropout),
        )
        self.channel_mixing = nn.Sequential(
            nn.Conv1d(channels, n_features * d_ff, 1, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_features * d_ff, channels, 1, groups=d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, n_features, d_model, patches = x.shape
        x = self.depthwise(x.reshape(batch_size, n_features * d_model, patches))
        x = self.norm(x.reshape(batch_size * n_features, d_model, patches))
        x = x.reshape(batch_size, n_features, d_model, patches)
        x = self.feature_mixing(x.reshape(batch_size, n_features * d_model, patches))
        x = x.reshape(batch_size, n_features, d_model, patches)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, d_model * n_features, patches)
        x = self.channel_mixing(x)
        x = x.reshape(batch_size, d_model, n_features, patches).permute(0, 2, 1, 3)
        return residual + x


class _FlattenHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        head_nf: int,
        target_steps: int,
        dropout: float,
        individual: bool,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.dropout = nn.Dropout(dropout)
        if individual:
            self.linears = nn.ModuleList([nn.Linear(head_nf, target_steps) for _ in range(n_features)])
        else:
            self.linear = nn.Linear(head_nf, target_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=-2)
        if self.individual:
            return torch.stack([self.dropout(layer(x[:, index, :])) for index, layer in enumerate(self.linears)], dim=1)
        return self.dropout(self.linear(x))


class _ModernTCNBackbone(nn.Module):
    def __init__(
        self,
        d_time: int,
        d_feature: int,
        d_model: int,
        ffn_ratio: int,
        num_blocks: int,
        large_size: int,
        small_size: int,
        patch_size: int,
        patch_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 1 <= patch_stride <= patch_size <= d_time:
            raise ValueError("ModernTCN requires 1 <= patch_stride <= patch_size <= d_time")
        if large_size % 2 == 0 or small_size % 2 == 0:
            raise ValueError("ModernTCN kernel sizes must be odd to preserve the sequence length")

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # forward pads the tail by pad_length before unfolding, so the unfold
        # window count — and therefore head_nf — is exactly this formula.
        self.pad_length = patch_size - patch_stride
        self.n_patches = (d_time + self.pad_length - patch_size) // patch_stride + 1
        self.stem = nn.Linear(patch_size, d_model)
        self.blocks = nn.ModuleList(
            [
                _ModernTCNBlock(
                    d_feature,
                    d_model,
                    ffn_ratio,
                    large_size,
                    small_size,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head_nf = d_model * self.n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, feature, time]
        if self.pad_length:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, self.pad_length)], dim=-1)
        x = self.stem(x.unfold(-1, self.patch_size, self.patch_stride)).permute(0, 1, 3, 2)
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCN(nn.Module):
    def __init__(
        self,
        d_feature: int,
        d_time: int,
        d_model: int,
        ffn_ratio: int,
        num_blocks: int,
        large_size: int,
        small_size: int,
        patch_size: int,
        patch_stride: int,
        dropout: float,
        head_dropout: float,
        individual: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.backbone = _ModernTCNBackbone(
            d_time,
            d_feature,
            d_model,
            ffn_ratio,
            num_blocks,
            large_size,
            small_size,
            patch_size,
            patch_stride,
            dropout,
        )
        self.head = _FlattenHead(
            d_feature,
            self.backbone.head_nf,
            d_time,
            head_dropout,
            individual,
        )

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        observed_mask = (1.0 - X_NA_mask).to(X_NA.dtype)
        encoded = self.backbone(X_NA.transpose(1, 2))
        reconstruction = self.head(encoded).transpose(1, 2)
        imputed_data = observed_mask * X_NA + (1.0 - observed_mask) * reconstruction

        loss = masked_mse_cal(reconstruction, X_NA, observed_mask)
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mse_cal(imputed_data, X_N, X_A_mask)
        return {"imp_loss": loss, "imp_data": imputed_data}
