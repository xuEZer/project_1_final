"""SAITS: Self-Attention-based Imputation for Time Series.

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

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.extern.SAITS.layers import EncoderLayer, PositionalEncoding
from src.models.extern.SAITS.utils import masked_mse_cal


class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        param_sharing_strategy,
        diagonal_attention_mask,
        device,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.param_sharing_strategy = param_sharing_strategy
        actual_d_feature = d_feature * 2

        # between_group shares one layer stack across forward passes;
        # inner_group follows ALBERT's group-wise sharing strategy.
        layers_per_block = (
            n_group_inner_layers
            if self.param_sharing_strategy == "between_group"
            else n_groups
        )

        def build_block():
            return nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        diagonal_attention_mask=diagonal_attention_mask,
                        device=device,
                    )
                    for _ in range(layers_per_block)
                ]
            )

        self.layer_stack_for_first_block = build_block()
        self.layer_stack_for_second_block = build_block()

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, X, X_mask):
        masks = X_mask
        input_X_for_first = torch.cat([X, masks], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        attn_weights = attn_weights.squeeze(dim=1)
        if len(attn_weights.shape) == 4:
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, X_NA, X_NA_mask, X_N=None, X_A_mask=None):
        X, masks = X_NA, 1 - X_NA_mask
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X, masks)

        reconstruction_loss = (
            masked_mse_cal(X_tilde_1, X, masks)
            + masked_mse_cal(X_tilde_2, X, masks)
            + masked_mse_cal(X_tilde_3, X, masks)
        ) / 3

        loss = reconstruction_loss
        if X_N is not None and X_A_mask is not None:
            loss += masked_mse_cal(X_tilde_3, X_N, X_A_mask)

        return {
            "imp_loss": loss,
            "imp_data": imputed_data,
        }
