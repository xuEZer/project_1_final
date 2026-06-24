"""Sundial predictor wrapper — inference-only interface for the JointModel.

Wraps ``thuml/sundial-base-128m`` (or variants) from HuggingFace so the
JointModel can call it as a pure inference module.  No loss computation,
no backpropagation.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class SundialPredictor(nn.Module):
    """Sundial forecasting model — inference-only.

    Uses HuggingFace ``AutoModelForCausalLM`` under the hood.  The
    ``forward`` call runs ``model.generate()``, which does **not**
    build a gradient graph.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"thuml/sundial-base-128m"``.
    num_samples : int
        Number of trajectories to sample per input sequence.
    sample_aggregation : str
        How to collapse the sample dimension.  ``"mean"`` averages
        across samples; other values return the raw samples.
    forecast_length : int
        Number of future steps to generate (pred_len from dataset config).
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model_name: str,
        num_samples: int,
        sample_aggregation: str,
        forecast_length: int,
        device: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_samples = num_samples
        self.sample_aggregation = sample_aggregation
        self.forecast_length = forecast_length
        self.device = device

        logger.info(f"Loading Sundial model '{model_name}' (inference-only) ...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self.backbone.eval()
        logger.info("Sundial model loaded.")

    def forward(
        self,
        imputed_X: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate forecasts from imputed lookback windows.

        Parameters
        ----------
        imputed_X : torch.Tensor
            Imputed lookback sequence, shape ``(B, T, D)``.

        Returns
        -------
        dict
            ``{"pred_data": torch.Tensor}`` with shape ``(B, P, D)``
            where ``P = forecast_length``.
        """
        B, T, D = imputed_X.shape

        # Sundial is a univariate foundation model; for multivariate
        # data we generate forecasts per-feature and stack.
        pred_list: list[torch.Tensor] = []

        with torch.no_grad():
            for feat_idx in range(D):
                seqs = imputed_X[:, :, feat_idx]  # (B, T)

                output = self.backbone.generate(
                    seqs,
                    max_new_tokens=self.forecast_length,
                    num_samples=self.num_samples,
                )
                # output shape: (B, num_samples, forecast_length)

                if self.sample_aggregation == "mean":
                    output = output.mean(dim=1, keepdim=False)  # (B, P)

                pred_list.append(output)  # each (B, P)

        # Stack features back: (B, P, D)
        pred_data = torch.stack(pred_list, dim=-1)

        return {"pred_data": pred_data}
