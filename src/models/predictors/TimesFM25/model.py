"""TimesFM 2.5 predictor wrapper — inference-only interface for the JointModel.

Wraps ``google/timesfm-2.5-200m-pytorch`` so the JointModel can call it as
a pure inference module.  No loss computation, no backpropagation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import timesfm

logger = logging.getLogger(__name__)


class TimesFM25Predictor(nn.Module):
    """TimesFM 2.5 forecasting model — inference-only.

    Uses the ``timesfm`` library under the hood.  The ``forward`` call
    delegates to ``model.forecast()``, which does **not** build a gradient
    graph.

    Parameters
    ----------
    model_name : str
        HuggingFace / model id, e.g. ``"google/timesfm-2.5-200m-pytorch"``.
    forecast_length : int
        Number of future steps to generate (pred_len from dataset config).
    max_context : int
        Maximum context length accepted by the model.
    per_core_batch_size : int
        Sequences batched per core (derived from ``training.batch_size``).
    normalize_inputs : bool
        Whether the model normalises inputs internally.
    use_continuous_quantile_head : bool
        Use continuous quantile head for probabilistic forecasts.
    force_flip_invariance : bool
        Enforce flip-invariance in the forecast.
    infer_is_positive : bool
        Infer if the input series is strictly positive.
    fix_quantile_crossing : bool
        Fix crossing quantiles in probabilistic output.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        max_context: int,
        per_core_batch_size: int,
        normalize_inputs: bool,
        use_continuous_quantile_head: bool,
        force_flip_invariance: bool,
        infer_is_positive: bool,
        fix_quantile_crossing: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.forecast_length = forecast_length
        self.device = device

        torch_compile = device.startswith("cuda")

        logger.info(
            f"Loading TimesFM 2.5 model '{model_name}' (inference-only, "
            f"compile={torch_compile}, per_core_bs={per_core_batch_size}) ..."
        )
        self.backbone = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_name, torch_compile=torch_compile
        )
        self.backbone.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=forecast_length,
                per_core_batch_size=per_core_batch_size,
                normalize_inputs=normalize_inputs,
                use_continuous_quantile_head=use_continuous_quantile_head,
                force_flip_invariance=force_flip_invariance,
                infer_is_positive=infer_is_positive,
                fix_quantile_crossing=fix_quantile_crossing,
            )
        )
        logger.info("TimesFM 2.5 model loaded and compiled.")

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

        # TimesFM is univariate — flatten batch × features into a list of
        # 1-D sequences.
        inputs = [
            imputed_X[b, :, d].cpu().float().numpy()
            for b in range(B)
            for d in range(D)
        ]

        point_forecast, _ = self.backbone.forecast(
            horizon=self.forecast_length,
            inputs=inputs,
        )
        # point_forecast: (B*D, P)

        pred_data = (
            torch.from_numpy(point_forecast)
            .float()
            .to(self.device)
            .reshape(B, self.forecast_length, D)
        )

        return {"pred_data": pred_data}
