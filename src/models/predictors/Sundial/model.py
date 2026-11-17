"""Sundial predictor — inference-only forecasting via HuggingFace generate()."""

from __future__ import annotations

import logging
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM

from src.rng import torch_seed_scope

logger = logging.getLogger(__name__)


def _assert_transformers_4_41() -> None:
    """Require the Transformers generation interface used by this adapter."""
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
            f"Sundial requires transformers 4.41, but "
            f"{transformers.__version__} is installed. Install "
            f"transformers==4.41.* before loading Sundial."
        )


def _configure_generation(model: nn.Module) -> None:
    """Bind Sundial's time-series search to the Transformers 4.41 entry point."""
    _assert_transformers_4_41()
    sundial_search = model._greedy_search

    def _sample_forecast(
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

    model._sample = MethodType(_sample_forecast, model)  # type: ignore[attr-defined]


class SundialPredictor(nn.Module):
    """Sundial forecasting model (e.g. ``thuml/sundial-base-128m``).

    Uses :meth:`AutoModelForCausalLM.generate` for autoregressive sampling.
    Operates per-feature (univariate), then stacks results back to
    ``(B, P, D)``.
    """

    deterministic_window_queries = True

    def __init__(
        self,
        model_name: str,
        num_samples: int,
        revision: str,
        cache_dir: str,
        dtype: str,
        forecast_length: int,
        device: str,
        query_seed: int,
    ) -> None:
        super().__init__()
        self.forecast_length = forecast_length
        self.num_samples = num_samples
        self.query_seed = int(query_seed)
        if dtype != "float32":
            raise ValueError("Sundial queries require float32 precision")

        logger.info("Loading Sundial model '%s' ...", model_name)
        model_path = snapshot_download(
            model_name, revision=revision, cache_dir=cache_dir
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)
        _configure_generation(self.backbone)
        self.backbone.eval().requires_grad_(False)
        logger.info("Sundial model loaded on %s.", device)

    @torch.inference_mode()
    def forward(self, imputed_X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forecast each context with fixed sampling noise and independent variables."""
        if not torch.isfinite(imputed_X).all():
            raise ValueError("Sundial inputs must be finite")
        batch_size, context_length, features = imputed_X.shape
        input_token_len = int(self.backbone.config.input_token_len)
        if context_length % input_token_len != 0:
            raise ValueError(
                "Sundial context length must be divisible by "
                f"input_token_len={input_token_len}, got {context_length}."
            )
        attention_mask = torch.ones(
            (1, context_length // input_token_len),
            dtype=torch.long,
            device=imputed_X.device,
        )
        contexts = imputed_X.permute(0, 2, 1).reshape(-1, context_length)
        forecasts = []
        for context in contexts:
            series = context.unsqueeze(0)
            mean = series.mean(dim=-1, keepdim=True)
            stdev = series.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            # A fixed noise stream defines the same point forecast for every query.
            with torch_seed_scope(self.query_seed, "sundial_query"):
                output = self.backbone.generate(
                    (series - mean) / stdev,
                    max_new_tokens=self.forecast_length,
                    num_samples=self.num_samples,
                    attention_mask=attention_mask,
                    revin=False,
                )
            expected = (1, self.num_samples, self.forecast_length)
            if output.shape != expected:
                raise RuntimeError(
                    f"Expected Sundial sample shape {expected}, got {output.shape}"
                )
            point = output.mean(dim=1) * stdev + mean
            if not torch.isfinite(point).all():
                raise ArithmeticError("Sundial produced nonfinite forecasts")
            forecasts.append(point[0])
        pred_data = (
            torch.stack(forecasts)
            .reshape(batch_size, features, self.forecast_length)
            .permute(0, 2, 1)
            .contiguous()
        )
        return {"pred_data": pred_data}
