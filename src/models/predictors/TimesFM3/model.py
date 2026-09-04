"""TimesFM 3 predictor using Google's official PyTorch evaluator."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TimesFM3Predictor(nn.Module):
    """Frozen, zero-shot multivariate TimesFM 3 forecaster.

    Each FOTSI sample is passed as one multivariate context, preserving the
    cross-channel information that TimesFM 3 was trained to use.
    """

    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        device: str,
        per_core_batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.forecast_length = int(forecast_length)
        self.device = _resolve_device(device)

        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
        except ImportError as exc:
            raise ImportError(
                "TimesFM 3 requires the official `timesfm[torch]` package. "
                "Install the project dependencies with `uv sync`."
            ) from exc

        logger.info("Loading TimesFM 3 model '%s' ...", model_name)
        config = ModelConfig(
            checkpoint_path=model_name,
            per_core_batch_size=int(per_core_batch_size),
            device=self.device,
        )
        self.evaluator = TimesFM3Evaluator(config)
        logger.info("TimesFM 3 model loaded on %s.", self.device)

    def _apply(self, fn):
        """The evaluator owns device placement and has no module parameters."""
        return self

    def to(self, *args, **kwargs):
        """Keep the evaluator on the device selected at construction time."""
        return self

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return point forecasts with shape ``(B, forecast_length, D)``."""
        batch_size, _context_length, n_variates = imputed_X.shape
        contexts = [
            sample.T
            for sample in imputed_X.detach().float().cpu().numpy()
        ]
        outputs = list(
            self.evaluator.predict_batch(
                contexts=contexts,
                horizon=self.forecast_length,
                return_quantiles=False,
                use_symmetric_averaging=False,
                make_positive=False,
            )
        )
        if len(outputs) != batch_size:
            raise RuntimeError(
                f"TimesFM 3 returned {len(outputs)} forecasts for "
                f"a batch of {batch_size}."
            )

        forecasts = np.stack(
            [np.asarray(output.forecast, dtype=np.float32) for output in outputs]
        )
        expected = (batch_size, n_variates, self.forecast_length)
        if forecasts.shape != expected:
            raise RuntimeError(
                f"Expected TimesFM 3 forecast shape {expected}, got "
                f"{forecasts.shape}."
            )
        if not np.isfinite(forecasts).all():
            n_bad = int(forecasts.size - np.isfinite(forecasts).sum())
            raise RuntimeError(
                f"TimesFM 3 produced {n_bad} non-finite forecast(s)."
            )

        pred_data = torch.from_numpy(forecasts).permute(0, 2, 1).to(
            device=imputed_X.device, dtype=torch.float32,
        ).contiguous()
        return {"pred_data": pred_data}


def _resolve_device(device: str) -> str:
    device = str(device).strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device
