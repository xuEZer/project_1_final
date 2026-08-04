"""Chronos predictor — inference-only forecasting via the chronos package."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from chronos import ChronosBoltPipeline, ChronosPipeline

try:
    from chronos import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None  # type: ignore[assignment]
from huggingface_hub import snapshot_download
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class ChronosPredictor(nn.Module):
    """Chronos forecasting model (e.g. ``amazon/chronos-2``).

    Uses the ``chronos`` package pipeline.  Output is converted to a
    deterministic point forecast via mean aggregation.
    """

    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        num_samples: int,
        device: str,
    ) -> None:
        super().__init__()
        self.forecast_length = int(forecast_length)
        self.num_samples = int(num_samples)
        self.device = _resolve_device(device)

        logger.info("Loading Chronos model '%s' ...", model_name)
        try:
            model_path = snapshot_download(
                model_name, allow_patterns=["config.json", "model.safetensors"],
                local_files_only=True,
            )
        except Exception:
            model_path = snapshot_download(
                model_name, allow_patterns=["config.json", "model.safetensors"],
            )
        config = AutoConfig.from_pretrained(model_path)
        arch = next(iter(getattr(config, "architectures", []) or []), "")
        if "Chronos2" in arch:
            if Chronos2Pipeline is None:
                raise RuntimeError(
                    "Chronos-2 requires chronos-forecasting>=2.0; "
                    "update the project environment before running experiments."
                )
            pipeline_cls: Any = Chronos2Pipeline
        elif "ChronosBolt" in arch:
            pipeline_cls = ChronosBoltPipeline
        else:
            pipeline_cls = ChronosPipeline

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.pipeline = pipeline_cls.from_pretrained(
            model_path, device_map=self.device, torch_dtype=dtype,
        )
        self._accepts_num_samples = (
            "num_samples" in inspect.signature(self.pipeline.predict).parameters
        )
        logger.info("Chronos model loaded on %s.", self.device)

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate forecasts from imputed lookback windows.

        Args:
            imputed_X: ``(B, T, D)`` imputed time series.

        Returns:
            ``{"pred_data": (B, forecast_length, D)}``.
        """
        B, _T, D = imputed_X.shape

        if not torch.isfinite(imputed_X).all():
            n_bad = int((~torch.isfinite(imputed_X)).sum().item())
            logger.warning("Chronos: sanitizing %d non-finite input(s).", n_bad)
            imputed_X = torch.nan_to_num(imputed_X, nan=0.0, posinf=0.0, neginf=0.0)

        # (B, T, D) → (B*D, T) on CPU for the pipeline
        contexts = (
            imputed_X.detach()
            .permute(0, 2, 1)
            .reshape(B * D, -1)
            .to(device="cpu", dtype=torch.float32)
        )

        predict_kwargs: dict[str, Any] = {
            "prediction_length": self.forecast_length,
        }
        if self._accepts_num_samples:
            predict_kwargs["num_samples"] = self.num_samples

        # Chronos-2 requires (n_series, n_variates, history_length), whereas
        # Chronos and Chronos-Bolt accept (n_series, history_length).  Keep
        # variables independent here to preserve the predictor's established
        # D=1 evaluation semantics.
        pipeline_inputs = contexts
        if Chronos2Pipeline is not None and isinstance(self.pipeline, Chronos2Pipeline):
            pipeline_inputs = contexts.unsqueeze(1)

        if pipeline_inputs.device.type != "cpu":
            raise RuntimeError(
                "Chronos pipeline inputs must be CPU tensors before its "
                "internal pinned-memory DataLoader runs."
            )

        samples = self.pipeline.predict(pipeline_inputs, **predict_kwargs)
        point = _chronos_to_array(samples, self.forecast_length, batch_size=B * D)

        pred_data = (
            torch.from_numpy(point)
            .to(device=imputed_X.device, dtype=torch.float32)
            .reshape(B, D, self.forecast_length)
            .permute(0, 2, 1)
            .contiguous()
        )
        return {"pred_data": pred_data}


# ------------------------------------------------------------------ helpers --


def _resolve_device(device: str) -> str:
    """Normalize ``"auto"`` → ``"cuda"`` / ``"cpu"``."""
    device = str(device).strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def _chronos_to_array(
    output: Any, prediction_length: int, *, batch_size: int,
) -> np.ndarray:
    """Convert Chronos pipeline output → ``(batch_size, prediction_length)``."""
    if isinstance(output, (list, tuple)) and len(output) == batch_size:
        rows = [
            _forecast_to_array(item, prediction_length, batch_size=1)[0]
            for item in output
        ]
        return np.stack(rows, axis=0).astype(np.float32)
    return _forecast_to_array(output, prediction_length, batch_size=batch_size)


def _forecast_to_array(
    output: Any, prediction_length: int, *, batch_size: int,
) -> np.ndarray:
    """Normalize heterogeneous forecast backend output to ``(B, P)`` float32.

    Layout convention (verified against chronos-forecasting 2.3.1): every
    supported pipeline returns the horizon as the LAST dim —

    * ``ChronosPipeline.predict``   → tensor ``(B, num_samples, P)``
    * ``ChronosBoltPipeline.predict`` → tensor ``(B, n_quantiles, P)``
    * ``Chronos2Pipeline.predict``  → list of tensors ``(n_variates,
      n_quantiles, P)``

    The 3-D case is therefore resolved deterministically (horizon on dim -1)
    instead of guessing from dimension sizes, which silently mis-averages the
    samples axis whenever ``num_samples >= prediction_length``.
    """
    if isinstance(output, (tuple, list)) and output:
        output = output[0]
    if isinstance(output, torch.Tensor):
        arr = output.detach().float().cpu().numpy()
    else:
        arr = np.asarray(output, dtype=np.float32)

    if arr.ndim == 1:
        if batch_size != 1 or arr.shape[0] < prediction_length:
            raise RuntimeError(f"Unexpected Chronos output shape: {arr.shape}")
        arr = arr[-prediction_length:][None, :]
    elif arr.ndim == 2:
        if arr.shape[0] == batch_size and arr.shape[1] >= prediction_length:
            arr = arr[:, -prediction_length:]
        elif arr.shape[1] == batch_size and arr.shape[0] >= prediction_length:
            arr = arr[-prediction_length:, :].T
        else:
            raise RuntimeError(f"Unexpected Chronos output shape: {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[0] != batch_size:
            raise RuntimeError(f"Unexpected Chronos output shape: {arr.shape}")
        # (B, S, P) is the canonical layout: horizon is always dim -1.
        if arr.shape[-1] == prediction_length:
            arr = arr.mean(axis=1)[:, -prediction_length:]
        # (B, P, S) legacy layout — kept only for exotic backends that do not
        # follow the canonical convention; no supported pipeline uses it.
        elif arr.shape[1] == prediction_length:
            arr = arr.mean(axis=2)[:, -prediction_length:]
        else:
            raise RuntimeError(
                f"Cannot determine Chronos sample layout: {arr.shape}; "
                f"expected horizon {prediction_length} on dim -1 or dim 1."
            )
    else:
        raise RuntimeError(f"Unexpected Chronos output shape: {arr.shape}")

    if arr.shape != (batch_size, prediction_length):
        raise RuntimeError(
            f"Expected forecast shape {(batch_size, prediction_length)}, "
            f"got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        n_bad = int(arr.size - np.isfinite(arr).sum())
        raise RuntimeError(f"Chronos produced {n_bad} non-finite forecast(s).")
    return np.asarray(arr, dtype=np.float32)
