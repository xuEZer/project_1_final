"""Frozen Chronos-2 forecasts with independent univariate contexts."""

from __future__ import annotations

import logging

import torch
from chronos import Chronos2Pipeline
from huggingface_hub import snapshot_download
from torch import nn

logger = logging.getLogger(__name__)


class ChronosPredictor(nn.Module):
    """Average Chronos-2 forecast quantiles into one point forecast."""

    deterministic_window_queries = True

    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        device: str,
        revision: str,
        cache_dir: str,
        dtype: str,
    ) -> None:
        super().__init__()
        if dtype != "float32":
            raise ValueError("Chronos-2 queries require float32 precision")
        self.forecast_length = int(forecast_length)
        logger.info("Loading Chronos-2 model %s at %s", model_name, revision)
        model_path = snapshot_download(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["config.json", "model.safetensors"],
        )
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float32
        )
        self.model = self.pipeline.model
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map completed contexts ``(B, T, D)`` to forecasts ``(B, P, D)``."""
        batch_size, _, features = imputed_X.shape
        if not torch.isfinite(imputed_X).all():
            raise ValueError("Chronos-2 inputs must be finite")
        # Each feature forms one independent task; the pipeline owns GPU transfers.
        contexts = (
            imputed_X.detach()
            .permute(0, 2, 1)
            .reshape(batch_size * features, 1, -1)
            .to(device="cpu", dtype=torch.float32)
        )
        forecasts = self.pipeline.predict(
            contexts,
            prediction_length=self.forecast_length,
            batch_size=batch_size * features,
            cross_learning=False,
        )
        if len(forecasts) != batch_size * features:
            raise RuntimeError("Chronos-2 returned an unexpected number of forecasts")
        for forecast in forecasts:
            if (
                forecast.ndim != 3
                or forecast.shape[0] != 1
                or forecast.shape[-1] != self.forecast_length
            ):
                raise RuntimeError(
                    f"Unexpected Chronos-2 forecast shape: {forecast.shape}"
                )
        point = torch.stack([forecast[0].float().mean(dim=0) for forecast in forecasts])
        if not torch.isfinite(point).all():
            raise ArithmeticError("Chronos-2 produced nonfinite forecasts")
        pred_data = (
            point.to(imputed_X.device)
            .reshape(batch_size, features, self.forecast_length)
            .permute(0, 2, 1)
            .contiguous()
        )
        return {"pred_data": pred_data}
