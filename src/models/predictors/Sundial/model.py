"""Sundial predictor — inference-only forecasting via HuggingFace generate()."""

from __future__ import annotations

import logging

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


class SundialPredictor(nn.Module):
    """Sundial forecasting model (e.g. ``thuml/sundial-small-32m``).

    Uses :meth:`AutoModelForCausalLM.generate` for autoregressive sampling.
    Operates per-feature (univariate), then stacks results back to
    ``(B, P, D)``.
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
        self.forecast_length = forecast_length
        self.num_samples = num_samples
        self.sample_aggregation = sample_aggregation

        logger.info("Loading Sundial model '%s' ...", model_name)
        _patch_dynamic_cache()
        try:
            model_path = snapshot_download(model_name, local_files_only=True)
        except Exception:
            model_path = snapshot_download(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
        ).to(device)
        self.backbone.eval()
        logger.info("Sundial model loaded on %s.", device)

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate forecasts from imputed lookback windows.

        Args:
            imputed_X: ``(B, T, D)`` imputed time series.

        Returns:
            ``{"pred_data": (B, forecast_length, D)}``.
        """
        B, _T, D = imputed_X.shape
        pred_list: list[torch.Tensor] = []

        for feat_idx in range(D):
            seqs = imputed_X[:, :, feat_idx]  # (B, T)
            output = self.backbone.generate(
                seqs,
                max_new_tokens=self.forecast_length,
                num_samples=self.num_samples,
            )  # (B, num_samples, forecast_length)

            if self.sample_aggregation == "mean":
                output = output.mean(dim=1)  # (B, forecast_length)

            pred_list.append(output)

        pred_data = torch.stack(pred_list, dim=-1)  # (B, forecast_length, D)
        return {"pred_data": pred_data}
