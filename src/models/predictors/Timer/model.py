"""Timer predictor — inference-only forecasting via HuggingFace AutoModel."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _patch_dynamic_cache() -> None:
    """Patch older remote-code models against newer Transformers cache APIs."""
    try:
        from transformers import DynamicCache
    except ImportError:
        return

    if not hasattr(DynamicCache, "seen_tokens"):

        def seen_tokens(self) -> int:  # type: ignore[no-redef]
            if hasattr(self, "get_seq_length"):
                return int(self.get_seq_length())
            return 0

        DynamicCache.seen_tokens = property(seen_tokens)  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_max_length") and hasattr(
        DynamicCache, "get_max_cache_shape"
    ):

        def get_max_length(self) -> int | None:  # type: ignore[no-redef]
            value = self.get_max_cache_shape()
            if isinstance(value, (tuple, list)):
                return None if not value else int(value[-1])
            return None if value is None else int(value)

        DynamicCache.get_max_length = get_max_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(  # type: ignore[no-redef]
            self, new_seq_length: int, *_args, **_kwargs,
        ) -> int:
            prev = (
                int(self.get_seq_length())
                if hasattr(self, "get_seq_length")
                else 0
            )
            max_len = (
                self.get_max_length()
                if hasattr(self, "get_max_length")
                else None
            )
            if max_len is None:
                return prev
            return min(prev, max(0, int(max_len) - int(new_seq_length)))

        DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]


class TimerPredictor(nn.Module):
    """Timer forecasting model (e.g. ``thuml/timer-base-84m``).

    Uses :class:`AutoModelForCausalLM` with ``trust_remote_code=True``.
    Supports optional input normalization and per-batch CPU fallback for
    non-finite CUDA logits.
    """

    def __init__(
        self,
        model_name: str,
        forecast_length: int,
        normalize_inputs: bool,
        device: str,
        inference_device: str | None = None,
    ) -> None:
        super().__init__()
        self.forecast_length = int(forecast_length)
        self.normalize_inputs = bool(normalize_inputs)
        self.device = _resolve_device(inference_device or device)

        logger.info("Loading Timer model '%s' ...", model_name)
        _patch_dynamic_cache()
        try:
            model_path = snapshot_download(
                model_name,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "configuration_timer.py",
                    "modeling_timer.py",
                    "ts_generation_mixin.py",
                    "model.safetensors",
                ],
                local_files_only=True,
            )
        except Exception:
            model_path = snapshot_download(
                model_name,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "configuration_timer.py",
                    "modeling_timer.py",
                    "ts_generation_mixin.py",
                    "model.safetensors",
                ],
            )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
        )
        _move_module(self.backbone, self.device)
        self.backbone.eval()
        self.input_dtype = _model_input_dtype(self.backbone, self.device)
        logger.info("Timer model loaded on %s.", self.device)

    # -- keep Timer pinned to its inference device ---------------------------

    def _apply(self, fn):
        """Prevent parent ``.to()`` from moving Timer off its device."""
        return self

    def to(self, *args, **kwargs):
        """Keep Timer on its configured inference device."""
        _move_module(self.backbone, self.device)
        self.input_dtype = _model_input_dtype(self.backbone, self.device)
        return self

    # -- forward pass ---------------------------------------------------------

    def _forward_logits(
        self,
        contexts: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run Timer backbone; return ``(logits, mean, std)``.

        *mean* and *std* must be computed on the same data passed as
        *contexts* (i.e. AFTER :func:`_patch_length`), so that
        normalisation and denormalisation use consistent statistics.
        """
        model_input = contexts
        if mean is not None and std is not None:
            model_input = (contexts - mean) / std

        output = self.backbone(model_input, max_output_length=self.forecast_length)
        raw = output.logits if hasattr(output, "logits") else output
        return raw, mean, std

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
            raise RuntimeError(f"Timer received {n_bad} non-finite input(s).")

        contexts = (
            imputed_X.detach()
            .permute(0, 2, 1)
            .reshape(B * D, -1)
            .to(device=self.device, dtype=self.input_dtype)
        )

        contexts = _patch_length(contexts, self.backbone)

        if self.normalize_inputs:
            mean = contexts.mean(dim=-1, keepdim=True)
            std = contexts.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        else:
            mean = std = None

        raw_logits, mean, std = self._forward_logits(contexts, mean=mean, std=std)

        # Timer's CUDA float16 logits occasionally go non-finite; retry the
        # batch once on CPU and move the module back so the configured device
        # and dtype stay untouched for subsequent calls.
        run_device = self.device
        if (
            isinstance(raw_logits, torch.Tensor)
            and not torch.isfinite(raw_logits).all()
        ):
            n_bad = int((~torch.isfinite(raw_logits)).sum().item())
            if self.device != "cpu":
                logger.warning(
                    "Timer produced %d non-finite logit(s) on %s; "
                    "retrying on CPU.", n_bad, self.device,
                )
                run_device = "cpu"
                _move_module(self.backbone, run_device)
                try:
                    run_dtype = torch.float32
                    contexts = contexts.to(device=run_device, dtype=run_dtype)
                    if mean is not None:
                        mean = mean.to(device=run_device, dtype=run_dtype)
                    if std is not None:
                        std = std.to(device=run_device, dtype=run_dtype)
                    raw_logits, mean, std = self._forward_logits(
                        contexts, mean=mean, std=std,
                    )
                finally:
                    _move_module(self.backbone, self.device)
            else:
                raise RuntimeError(
                    f"Timer produced {n_bad} non-finite logit(s) even on CPU."
                )

        point = _logits_to_array(raw_logits, self.forecast_length, B * D)
        pred = torch.from_numpy(point).to(device=run_device, dtype=torch.float32)
        if mean is not None and std is not None:
            pred = pred * std.float() + mean.float()

        pred_data = (
            pred.to(device=imputed_X.device, dtype=torch.float32)
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


def _model_input_dtype(module: nn.Module, device: str) -> torch.dtype:
    """Pick float dtype matching the model's parameter dtype."""
    try:
        for param in module.parameters():
            if param.is_floating_point():
                if device.startswith("cuda") and param.dtype in {
                    torch.float16, torch.bfloat16, torch.float32,
                }:
                    return param.dtype
                return torch.float32
    except Exception:
        pass
    return torch.float32


def _move_module(module: nn.Module, device: str) -> None:
    """Move *module* to *device*, tolerating custom ``to`` signatures."""
    try:
        module.to(device)
    except TypeError:
        module.to(torch.device(device))


def _patch_length(contexts: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Trim oldest timesteps so the sequence length is a multiple of
    ``input_token_len``.

    Left-padding distorts the temporal pattern and the Timer remote code
    does not accept a standard HuggingFace ``attention_mask`` to suppress
    padded positions.  Trimming the oldest values avoids both issues at
    the cost of discarding up to ``input_token_len - 1`` context steps.
    """
    token_len = getattr(getattr(model, "config", None), "input_token_len", None)
    if token_len is None:
        return contexts
    token_len = int(token_len)
    if token_len <= 1:
        return contexts
    remainder = contexts.shape[1] % token_len
    if remainder == 0:
        return contexts
    return contexts[:, remainder:]


def _logits_to_array(
    raw: torch.Tensor, prediction_length: int, batch_size: int,
) -> np.ndarray:
    """Convert Timer raw logits → ``(batch_size, prediction_length)`` float32."""
    arr = raw.detach().float().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[-prediction_length:][None, :]
    elif arr.ndim == 2:
        if arr.shape[0] == batch_size and arr.shape[1] >= prediction_length:
            arr = arr[:, -prediction_length:]
        elif arr.shape[1] == batch_size and arr.shape[0] >= prediction_length:
            arr = arr[-prediction_length:, :].T
        else:
            raise RuntimeError(f"Unexpected Timer logit shape: {arr.shape}")
    else:
        raise RuntimeError(f"Unexpected Timer logit shape: {arr.shape}")

    if arr.shape != (batch_size, prediction_length):
        raise RuntimeError(
            f"Expected forecast shape {(batch_size, prediction_length)}, "
            f"got {arr.shape}."
        )
    return np.asarray(arr, dtype=np.float32)
