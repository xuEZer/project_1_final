"""BRITS imputer implementation used in the training pipeline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.extern.SAITS.utils import masked_mse_cal


class FeatureRegression(nn.Module):
    """Feature-wise regression layer with masked self-connections."""

    def __init__(self, input_size):
        """Initialize projection parameters and masks."""
        super().__init__()
        self.build(input_size)

    def build(self, input_size):
        """Create trainable parameters and fixed connection mask."""
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable parameters with uniform initialization."""
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """Apply masked linear regression over feature dimension."""
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class TemporalDecay(nn.Module):
    """Temporal decay module that maps time gaps to decay factors."""

    def __init__(self, input_size, output_size, diag=False):
        """Initialize decay projection with optional diagonal constraints."""
        super().__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        """Create trainable parameters and optional diagonal mask."""
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset trainable parameters with uniform initialization."""
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        """Compute exponential decay factors from time deltas."""
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class FeatureGenerator(nn.Module):
    """Generate per-timestep time-gap features from observation masks."""

    def __init__(self):
        """Initialize feature generator."""
        super().__init__()

    def forward(self, x, mask):
        """Compute cumulative deltas for each feature over time."""
        batch_size, seq_len, num_features = x.shape
        deltas = torch.zeros_like(x).to(x.device)

        for t in range(1, seq_len):
            deltas[:, t, :] = 1 + deltas[:, t - 1, :] * (1 - mask[:, t - 1, :])

        return deltas


class RITS(nn.Module):
    """Recurrent imputation block used in one temporal direction."""

    def __init__(self, d_feature, hidden_size):
        """Initialize recurrent cell and auxiliary regression layers."""
        super().__init__()
        self.d_feature = d_feature
        self.hidden_size = hidden_size

        self.rnn_cell = nn.LSTMCell(d_feature * 2, hidden_size)

        self.temp_decay_h = TemporalDecay(
            input_size=d_feature, output_size=hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=d_feature, output_size=d_feature, diag=True
        )

        self.hist_reg = nn.Linear(hidden_size, d_feature)
        self.feat_reg = FeatureRegression(d_feature)

        self.weight_combine = nn.Linear(d_feature * 2, d_feature)

    def forward(self, values, masks, deltas):
        """Impute a sequence and return reconstruction loss and outputs."""
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
            loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            alpha = torch.sigmoid(alpha)

            c_h = alpha * z_h + (1 - alpha) * x_h
            loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim=1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(1))

        imputations = torch.cat(imputations, dim=1)

        return loss / seq_len, imputations


class BRITS(nn.Module):
    """Bidirectional RITS-based imputer."""

    def __init__(self, d_feature, d_model, device):
        """Initialize forward and backward RITS modules."""
        super().__init__()
        self.device = device
        self.rits_f = RITS(d_feature, d_model)
        self.rits_b = RITS(d_feature, d_model)
        self.feat_gen = FeatureGenerator()

    def forward(self, X_NA, X_NA_mask, X_N=None, X_A_mask=None):
        """Run bidirectional imputation and compute reconstruction loss."""
        masks = 1 - X_NA_mask
        values = X_NA

        deltas_f = self.feat_gen(values, masks)

        values_b = torch.flip(values, [1])
        masks_b = torch.flip(masks, [1])
        deltas_b = self.feat_gen(values_b, masks_b)

        loss_f, imp_f = self.rits_f(values, masks, deltas_f)
        loss_b, imp_b = self.rits_b(values_b, masks_b, deltas_b)

        imp_b = torch.flip(imp_b, [1])

        imputations = (imp_f + imp_b) / 2
        const_loss = torch.mean(torch.pow(imp_f - imp_b, 2))
        loss = loss_f + loss_b + 0.1 * const_loss

        if X_N is not None and X_A_mask is not None:
            imp_loss = masked_mse_cal(imputations, X_N, X_A_mask)
            loss += imp_loss

        return {"imp_loss": loss, "imp_data": imputations}
