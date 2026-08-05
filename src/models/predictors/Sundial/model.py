"""Sundial predictor — inference-only forecasting via HuggingFace generate()."""

from __future__ import annotations

import logging
from types import MethodType

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


def _assert_transformers_4_41() -> None:
    """Raise unless the installed Transformers matches the patched signature.

    ``_patch_sundial_generation`` adapts the Transformers 4.41 ``_sample``
    signature.  Against any other version it silently drops sampling options
    (e.g. ``top_k``/``top_p``) or misroutes the search, so the mismatch must
    fail loudly at model load time.
    """
    import transformers

    parts = transformers.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        raise RuntimeError(
            f"Cannot parse transformers version {transformers.__version__!r}."
        ) from None
    if (major, minor) != (4, 41):
        raise RuntimeError(
            f"Sundial's generation patch targets transformers 4.41, but "
            f"{transformers.__version__} is installed. Install "
            f"transformers==4.41.* (or migrate the patch) before Sundial."
        )


def _patch_sundial_generation(model: nn.Module) -> None:
    """Route the Transformers 4.41 generation entry point to Sundial's search.

    Sundial's remote code overrides ``_greedy_search``.  Transformers 4.41
    merged greedy search into ``_sample``, bypassing that override and
    returning a standard 2-D causal-LM sequence.  Adapt the new signature to
    the remote implementation without modifying the HuggingFace cache.
    """
    _assert_transformers_4_41()
    sundial_search = getattr(model, "_greedy_search", None)
    if sundial_search is None:
        return

    def _sample_compat(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        logits_warper=None,
        **model_kwargs,
    ):
        del self, logits_warper
        return sundial_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            output_scores=generation_config.output_scores,
            output_logits=getattr(generation_config, "output_logits", None),
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    model._sample = MethodType(_sample_compat, model)  # type: ignore[attr-defined]


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
        _patch_sundial_generation(self.backbone)
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
            input_token_len = int(self.backbone.config.input_token_len)
            if seqs.shape[1] % input_token_len != 0:
                raise ValueError(
                    "Sundial context length must be divisible by "
                    f"input_token_len={input_token_len}, got {seqs.shape[1]}."
                )
            # Newer Transformers versions otherwise infer a timestep-level
            # mask of length T, while Sundial attention operates on T / patch
            # tokens.  Supplying the token-level mask keeps remote model code
            # compatible with the Chronos-2 dependency floor.
            attention_mask = torch.ones(
                (B, seqs.shape[1] // input_token_len),
                dtype=torch.long,
                device=seqs.device,
            )
            # Sundial's remote TSGenerationMixin assumes that Transformers
            # returns samples as (B, S, P).  Recent Transformers releases call
            # the model's custom search routine, which returns (S, B, P)
            # instead.  The remote mixin therefore fails while broadcasting
            # its ReVIN statistics whenever B != S.  Disable remote ReVIN and
            # perform the same normalization locally after canonicalizing the
            # generated tensor layout.
            means = seqs.mean(dim=-1, keepdim=True)
            stdev = seqs.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            normalized_seqs = (seqs - means) / stdev
            output = self.backbone.generate(
                normalized_seqs,
                max_new_tokens=self.forecast_length,
                num_samples=self.num_samples,
                attention_mask=attention_mask,
                revin=False,
            )

            output = _canonicalize_samples(
                output,
                batch_size=B,
                num_samples=self.num_samples,
            )
            output = output * stdev.unsqueeze(1) + means.unsqueeze(1)

            # A truncated sequence would silently produce a shorter forecast;
            # fail instead of slicing past the generated horizon.
            if output.shape[-1] < self.forecast_length:
                raise RuntimeError(
                    f"Sundial returned {output.shape[-1]} generated steps, "
                    f"fewer than forecast_length={self.forecast_length}."
                )
            # Some Transformers versions return context + continuation for
            # this remote-code model, while older versions returned only the
            # continuation.  Normalize both forms explicitly.
            output = output[..., -self.forecast_length:]

            if self.sample_aggregation == "mean":
                output = output.mean(dim=1)  # (B, forecast_length)

            pred_list.append(output)

        pred_data = torch.stack(pred_list, dim=-1)  # (B, forecast_length, D)
        return {"pred_data": pred_data}


def _canonicalize_samples(
    output: torch.Tensor,
    *,
    batch_size: int,
    num_samples: int,
) -> torch.Tensor:
    """Return Sundial samples in the stable ``(B, S, P)`` layout."""
    if output.ndim == 2:
        if output.shape[0] != batch_size:
            raise RuntimeError(
                "Unexpected 2-D Sundial output shape "
                f"{tuple(output.shape)} for B={batch_size}, S={num_samples}."
            )
        return output.unsqueeze(1)

    if output.ndim != 3:
        raise RuntimeError(
            f"Expected a 2-D or 3-D Sundial output, got {tuple(output.shape)}."
        )
    if output.shape[:2] == (batch_size, num_samples):
        return output
    if output.shape[:2] == (num_samples, batch_size):
        return output.permute(1, 0, 2).contiguous()
    raise RuntimeError(
        "Unexpected Sundial sample layout "
        f"{tuple(output.shape)} for B={batch_size}, S={num_samples}."
    )
