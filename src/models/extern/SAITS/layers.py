"""Layer modules for self-attention models (Transformer and SAITS).

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

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention module."""

    def __init__(self, temperature, attn_dropout=0.1):
        """Initialize scaled dot-product attention.

        Args:
            temperature: Scale factor applied to query-key dot products.
            attn_dropout: Dropout rate on normalized attention weights.

        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        """Compute attention outputs from query, key, and value tensors.

        Args:
            q: Query tensor with shape [batch, head, len_q, d_k].
            k: Key tensor with shape [batch, head, len_k, d_k].
            v: Value tensor with shape [batch, head, len_v, d_v].
            attn_mask: Optional attention mask where 1 means masked.

        Returns:
            A tuple of attention output and attention weights.

        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, torch.finfo(attn.dtype).min)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention used by Transformer-style encoder layers."""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        """Initialize projection layers and attention core.

        Args:
            n_head: Number of attention heads.
            d_model: Model hidden dimension.
            d_k: Per-head key/query dimension.
            d_v: Per-head value dimension.
            attn_dropout: Dropout rate for attention probabilities.

        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        """Apply multi-head attention and project outputs back to model space.

        Args:
            q: Query tensor with shape [batch, len_q, d_model].
            k: Key tensor with shape [batch, len_k, d_model].
            v: Value tensor with shape [batch, len_v, d_model].
            attn_mask: Optional time-step mask for attention.

        Returns:
            A tuple containing projected output and attention weights.

        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network with pre-layer normalization."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        """Create two-layer feed-forward block.

        Args:
            d_in: Input and output feature dimension.
            d_hid: Hidden dimension of the intermediate layer.
            dropout: Dropout rate applied to block output.

        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Run feed-forward transformation with residual connection.

        Args:
            x: Input tensor with shape [batch, length, d_in].

        Returns:
            Transformed tensor with the same shape as input.

        """
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    """Single encoder layer composed of attention and feed-forward blocks."""

    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        **kwargs,
    ):
        """Initialize an encoder layer for temporal modeling.

        Args:
            d_time: Number of time steps.
            d_feature: Number of input features.
            d_model: Model hidden dimension.
            d_inner: Hidden dimension in feed-forward block.
            n_head: Number of attention heads.
            d_k: Per-head key/query dimension.
            d_v: Per-head value dimension.
            dropout: Dropout rate for residual outputs.
            attn_dropout: Dropout rate in attention probabilities.
            **kwargs: Extra arguments including diagonal_attention_mask and device.

        """
        super().__init__()

        self.diagonal_attention_mask = kwargs["diagonal_attention_mask"]
        self.device = kwargs["device"]
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        """Encode one layer with optional diagonal self-attention mask.

        Args:
            enc_input: Input tensor with shape [batch, time, d_model].

        Returns:
            A tuple of encoded tensor and attention weights.

        """
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module."""

    def __init__(self, d_hid, n_position=200):
        """Build and register sinusoidal encoding table.

        Args:
            d_hid: Hidden dimension used by the model.
            n_position: Maximum sequence length for position table.

        """
        super().__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Create sinusoidal position encoding table.

        Args:
            n_position: Maximum sequence length.
            d_hid: Hidden dimension of position encoding.

        Returns:
            A tensor with shape [1, n_position, d_hid].

        """

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """Add positional embeddings to input sequence.

        Args:
            x: Input tensor with shape [batch, length, d_hid].

        Returns:
            Input tensor with positional encoding added.

        """
        return x + self.pos_table[:, : x.size(1)].clone().detach()
