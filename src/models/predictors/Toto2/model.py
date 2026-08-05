"""Toto2 predictor — inference-only forecasting via Toto2Model from Datadog.

Toto-2.0 is a pretrained time-series foundation model that produces
quantile forecasts.  We use the median (0.5 quantile) as the point
forecast consumed by the rest of the FOTSI pipeline.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Toto2Predictor(nn.Module):
    """Toto-2.0 forecasting model (e.g. ``Datadog/Toto-2.0-4m``).

    Uses :class:`toto2.Toto2Model` for quantile forecasting at levels
    ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.  The median
    (index 4) is extracted as the deterministic point forecast.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        device: str,
    ) -> None:
        super().__init__()
        self.forecast_length = int(forecast_length)
        self.device = device

        logger.info("Loading Toto2 model '%s' ...", model_name)
        from toto2 import Toto2Model

        self.backbone = Toto2Model.from_pretrained(model_name)
        self.backbone = self.backbone.to(device).eval()
        logger.info("Toto2 model loaded on %s.", device)

    # -- keep Toto2 pinned to its device -----------------------------------

    def _apply(self, fn):
        """Prevent parent ``.to()`` from moving Toto2 off its device."""
        return self

    def to(self, *args, **kwargs):
        """Keep Toto2 on its configured device."""
        try:
            self.backbone.to(self.device)
        except TypeError:
            self.backbone.to(torch.device(self.device))
        return self

    # -- forward pass -------------------------------------------------------

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate forecasts from imputed lookback windows.

        Args:
            imputed_X: ``(B, T, D)`` imputed time series.

        Returns:
            ``{"pred_data": (B, forecast_length, D)}``.
        """
        B, _T, D = imputed_X.shape

        # Toto2 consumes (batch, n_variates, time_steps).
        target = imputed_X.permute(0, 2, 1).to(
            device=self.device, dtype=torch.float32,
        )
        target_mask = torch.ones_like(target, dtype=torch.bool)
        series_ids = (
            torch.arange(D, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # Quantiles are ordered ``[0.1, ..., 0.9]``; index 4 is the median,
        # which is the deterministic point forecast.
        quantiles = self.backbone.forecast(
            {
                "target": target,
                "target_mask": target_mask,
                "series_ids": series_ids,
            },
            horizon=self.forecast_length,
            decode_block_size=768,
            has_missing_values=False,
        )
        pred = quantiles[4]

        pred_data = pred.permute(0, 2, 1).to(
            device=imputed_X.device, dtype=torch.float32,
        ).contiguous()

        return {"pred_data": pred_data}
