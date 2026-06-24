"""DeepMVI adapter that exposes the project-standard imputer interface."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.extern.DeepMVI.model import OurModel
from src.models.extern.SAITS.utils import masked_mae_cal


class DeepMVI(nn.Module):
    """Adapter around DeepMVI's ``OurModel`` with project-compatible IO."""

    def __init__(
        self,
        d_feature: int,
        d_time: int,
        device: str,
        kernel_size: int = 10,
        block_size: int = 10,
        n_head: int = 2,
        use_context: bool = True,
        use_local: bool = True,
    ):
        """Initialize DeepMVI backbone and adapter options.

        Args:
            d_feature: Number of input features.
            d_time: Input sequence length.
            d_model: Hidden size placeholder kept for config compatibility.
            device: Runtime device string.
            kernel_size: DeepMVI temporal kernel size.
            block_size: Local context block size.
            n_head: Number of attention heads.
            use_context: Whether to enable contextual attention query.
            use_local: Whether to enable local context feature branch.

        """
        super().__init__()

        if use_local and (kernel_size <= block_size or kernel_size % block_size != 0):
            raise ValueError(
                "DeepMVI requires 0 < block_size < kernel_size and "
                "kernel_size % block_size == 0 when use_local=True"
            )

        self.d_feature = d_feature
        self.device = device
        self.backbone = OurModel(
            sizes=[1],
            kernel_size=kernel_size,
            block_size=block_size,
            time_len=d_time,
            nhead=n_head,
            use_embed=False,
            use_context=use_context,
            use_local=use_local,
        )

    def _build_context_info(
        self,
        flat_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Build minimal DeepMVI context tensors required by ``core``."""
        flat_batch_size, seq_len = flat_values.shape
        device = flat_values.device

        dummy_index = torch.zeros((flat_batch_size, 1), dtype=torch.long, device=device)
        dummy_tensor = torch.zeros((flat_batch_size, 1), device=device)
        # DeepMVI's AttentionModule constructs its base mask on CPU and adds this
        # bias before moving to value.device, so this tensor must stay on CPU.
        attention_bias = torch.zeros(
            (flat_batch_size, 1, seq_len),
            dtype=flat_values.dtype,
            device="cpu",
        )

        return [dummy_index, dummy_tensor, dummy_tensor, attention_bias]

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run DeepMVI imputation and return project-standard output fields."""
        masks = (1 - X_NA_mask).to(X_NA.dtype)

        batch_size, seq_len, d_feature = X_NA.shape
        if d_feature != self.d_feature:
            raise ValueError(
                f"Expected d_feature={self.d_feature}, but got input with {d_feature}"
            )

        flat_values = X_NA.permute(0, 2, 1).reshape(-1, seq_len)
        flat_masks = masks.permute(0, 2, 1).reshape(-1, seq_len)

        context_info = self._build_context_info(flat_values)
        residuals = torch.zeros_like(flat_values)

        pred_flat = self.backbone.core(
            flat_values,
            residuals,
            context_info,
            test=not self.training,
        )
        imputed_flat = flat_masks * flat_values + (1 - flat_masks) * pred_flat

        imputed_data = imputed_flat.reshape(batch_size, d_feature, seq_len).permute(
            0, 2, 1
        )

        reconstruction_loss = masked_mae_cal(imputed_data, X_NA, masks)
        loss = reconstruction_loss
        if X_N is not None and X_A_mask is not None:
            loss = loss + masked_mae_cal(imputed_data, X_N, X_A_mask)

        return {
            "imp_loss": loss,
            "imp_data": imputed_data,
        }
