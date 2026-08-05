"""Bidirectional recurrent imputer (BRITS) for time-series imputation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.extern.SAITS.utils import masked_mse_cal

CONSISTENCY_LOSS_WEIGHT = 0.1


def _uniform_init(*parameters: Parameter) -> None:
    for param in parameters:
        stdv = 1.0 / math.sqrt(param.size(0))
        param.data.uniform_(-stdv, stdv)


class FeatureRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        # Zero out the diagonal so a feature never regresses on itself.
        off_diagonal = torch.ones(input_size, input_size) - torch.eye(
            input_size, input_size
        )
        self.register_buffer("m", off_diagonal)
        _uniform_init(self.W, self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if diag:
            assert input_size == output_size
            self.register_buffer("m", torch.eye(input_size, input_size))
        _uniform_init(self.W, self.b)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        weight = self.W * self.m if self.diag else self.W
        return torch.exp(-F.relu(F.linear(d, weight, self.b)))


class FeatureGenerator(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        deltas = torch.zeros_like(x)
        for t in range(1, seq_len):
            deltas[:, t, :] = 1 + deltas[:, t - 1, :] * (1 - mask[:, t - 1, :])
        return deltas


class RITS(nn.Module):
    def __init__(self, d_feature: int, hidden_size: int):
        super().__init__()
        self.d_feature = d_feature
        self.hidden_size = hidden_size

        self.rnn_cell = nn.LSTMCell(d_feature * 2, hidden_size)
        self.temp_decay_h = TemporalDecay(d_feature, hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(d_feature, d_feature, diag=True)
        self.hist_reg = nn.Linear(hidden_size, d_feature)
        self.feat_reg = FeatureRegression(d_feature)
        self.weight_combine = nn.Linear(d_feature * 2, d_feature)

    def _observed_mae(
        self, target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(torch.abs(target - prediction) * mask) / (
            torch.sum(mask) + 1e-5
        )

    def forward(
        self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = values.size()

        h = torch.zeros((batch_size, self.hidden_size)).to(values.device)
        c = torch.zeros((batch_size, self.hidden_size)).to(values.device)

        imputations = []
        loss = 0.0

        for t in range(seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            loss += self._observed_mae(x, x_h, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            loss += self._observed_mae(x, z_h, m)

            alpha = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            loss += self._observed_mae(x, c_h, m)

            c_c = m * x + (1 - m) * c_h

            h, c = self.rnn_cell(torch.cat([c_c, m], dim=1), (h, c))
            imputations.append(c_c.unsqueeze(1))

        return loss / seq_len, torch.cat(imputations, dim=1)


class BRITS(nn.Module):
    def __init__(self, d_feature: int, d_model: int, device: str):
        super().__init__()
        self.rits_f = RITS(d_feature, d_model)
        self.rits_b = RITS(d_feature, d_model)
        self.feat_gen = FeatureGenerator()

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        masks = 1 - X_NA_mask

        deltas_f = self.feat_gen(X_NA, masks)

        values_b = torch.flip(X_NA, [1])
        masks_b = torch.flip(masks, [1])
        deltas_b = self.feat_gen(values_b, masks_b)

        loss_f, imp_f = self.rits_f(X_NA, masks, deltas_f)
        loss_b, imp_b = self.rits_b(values_b, masks_b, deltas_b)
        imp_b = torch.flip(imp_b, [1])

        imputations = (imp_f + imp_b) / 2
        consistency_loss = torch.mean(torch.pow(imp_f - imp_b, 2))
        loss = loss_f + loss_b + CONSISTENCY_LOSS_WEIGHT * consistency_loss

        if X_N is not None and X_A_mask is not None:
            loss += masked_mse_cal(imputations, X_N, X_A_mask)

        return {"imp_loss": loss, "imp_data": imputations}
