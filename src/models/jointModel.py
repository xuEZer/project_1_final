"""Joint training wrapper for imputation and prediction models.

The imputer is trained via its reconstruction / imputation loss.  The
predictor is **inference-only** — it runs inside ``torch.no_grad()``,
does not produce a loss, and never participates in backpropagation.
"""

from __future__ import annotations

import collections.abc
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.rng import derive_seed, torch_generator

optimizer_dict = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}


class JointModel(nn.Module):
    """Wrap an imputer (trainable) and predictor (inference-only)."""

    def __init__(
        self,
        imputer: nn.Module,
        predictor: nn.Module,
        optimizer_name: str,
        lr: float,
        device: str,
        tb_summary_writer: SummaryWriter,
        logger: logging.Logger,
    ) -> None:
        """Initialize the joint model and optimizer state.

        Args:
            imputer: Imputation module (trainable).
            predictor: Prediction module (inference-only, frozen).
            optimizer_name: Optimizer name (must be a key in
                ``optimizer_dict``).
            lr: Learning rate.
            device: Torch device string.
            tb_summary_writer: TensorBoard writer.
            logger: Logger instance.
        """
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor
        self.tb_summary_writer = tb_summary_writer
        self.logger = logger
        self.device = device

        self.global_epoch = 0
        self.global_optim_step = 0
        self.accum_steps = 1
        self.micro_batch_size = 1
        self.non_blocking_transfer = str(device).startswith("cuda")
        self.lr_scheduler = None
        self.base_seed = 0
        self.training_stochastic_seed = derive_seed(0, "training_stochastic_default")
        self.training_stochastic_seed_applied = False
        self.tip_generator = torch_generator(0, "tip_direction_default", device=device)

        # Only the imputer is trained — predictor is frozen
        self.optimizer = optimizer_dict[optimizer_name](
            self.imputer.parameters(), lr=lr
        )

        for p in self.imputer.parameters():
            p.requires_grad = True
        self.imputer.train()

        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()

    def set_predictor(self, predictor: nn.Module) -> None:
        """Replace the frozen predictor used for forecasting/evaluation."""
        self.predictor = predictor
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()

    def configure_random_generators(self, base_seed: int) -> None:
        """Configure independent Torch generators owned by this model."""
        self.base_seed = int(base_seed)
        self.training_stochastic_seed = derive_seed(base_seed, "training_stochastic")
        self.training_stochastic_seed_applied = False
        self.tip_generator = torch_generator(
            base_seed,
            "tip_direction",
            device=self.device,
        )
        self.logger.info(
            "Configured independent RNG: training_stochastic_seed=%d | "
            "tip_direction_seed=%d",
            self.training_stochastic_seed,
            self.tip_generator.initial_seed(),
        )

    def _seed_training_stochastic_ops(self) -> None:
        """Seed global Torch RNG only for ops without generator support."""
        if self.training_stochastic_seed_applied:
            return
        torch.manual_seed(self.training_stochastic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_stochastic_seed)
        self.training_stochastic_seed_applied = True

    def configure_lr_scheduler(
        self, scheduler_name: str, num_epoch: int, min_lr: float,
    ) -> None:
        """Configure an optional epoch-level learning-rate scheduler."""
        scheduler_name = scheduler_name.lower().strip()
        if scheduler_name in {"", "none", "off"}:
            self.lr_scheduler = None
            return
        if scheduler_name == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, num_epoch), eta_min=min_lr,
            )
            self.logger.info(
                "Cosine LR scheduler: lr=%.6g → %.6g over %d epochs",
                self.current_lr(), min_lr, max(1, num_epoch),
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    def current_lr(self) -> float:
        """Return the current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def forward(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None = None,
        X_A_mask: torch.Tensor | None = None,
        Y_N: torch.Tensor | None = None,
        Y_N_mask: torch.Tensor | None = None,
        run_predictor: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Run imputation and prediction forward passes.

        The predictor is called inside ``torch.no_grad()`` — no gradient
        graph is built through it.

        Args:
            X_NA: Input with natural + artificial missing values
                ``(B, T, D)``.
            X_NA_mask: Binary mask (1 = missing) ``(B, T, D)``.
            X_N: Ground-truth complete data ``(B, T, D)`` (optional —
                needed for artificial-missing loss during training).
            X_A_mask: Binary artificial-missing mask ``(B, T, D)``
                (optional).
            Y_N: Ground-truth future values ``(B, P, D)`` (optional).
            Y_N_mask: Binary mask for future values ``(B, P, D)``
                (optional, 1 = missing).
            run_predictor: Whether to run the frozen predictor. Training
                can skip it when only imputation loss is needed.

        Returns:
            Dictionary with keys:
            - ``imp_data``: Imputed time series ``(B, T, D)``.
            - ``imp_loss``: Imputation loss scalar (or ``None``).
            - ``pred_data``: Forecast from the imputed data
              ``(B, P, D)`` (or ``None`` if predictor returns it).
            - ``pred_loss``: Masked MSE prediction loss (or ``None``).
            - ``loss``: Total loss for backpropagation (same as
              ``imp_loss``).
        """
        imputer_output = self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)
        imputed_X = imputer_output["imp_data"]
        imp_loss = imputer_output["imp_loss"]

        # Predictor runs in inference mode — no gradient graph built
        pred_data = None
        if run_predictor:
            with torch.no_grad():
                predictor_output = self.predictor(imputed_X)
            pred_data = predictor_output.get("pred_data", None)

        # --- assertions ---
        assert not torch.isnan(imputed_X).any(), "Imputer output 'imp_data' contains NaN"
        if pred_data is not None:
            assert not torch.isnan(pred_data).any(), \
                "Predictor output 'pred_data' contains NaN"

        pred_loss_val = None
        if pred_data is not None and Y_N is not None:
            diff = (pred_data - Y_N) ** 2
            mask = 1.0 - Y_N_mask if Y_N_mask is not None else torch.ones_like(Y_N)
            pred_loss_val = (diff * mask).sum() / (mask.sum() + 1e-8)

        return {
            "imp_data": imputed_X,
            "imp_loss": imp_loss,
            "pred_data": pred_data,
            "pred_loss": pred_loss_val,
            "loss": imp_loss,
        }

    # ------------------------------------------------------------------
    # TIP helper methods
    # ------------------------------------------------------------------

    def _get_flattened_grad(self) -> torch.Tensor:
        """Flatten all imputer .grad tensors into a single 1-D vector."""
        grads: list[torch.Tensor] = []
        for p in self.imputer.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros_like(p).flatten())
        return torch.cat(grads)

    def _move_batch_to_device(
        self,
        batch: collections.abc.Iterable[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Move a dataloader batch to the configured device."""
        return tuple(
            t.to(self.device, non_blocking=self.non_blocking_transfer) for t in batch
        )

    def _set_flattened_grad(self, flat_grad: torch.Tensor) -> None:
        """Write a flat gradient vector back to imputer .grad attributes."""
        offset = 0
        for p in self.imputer.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset : offset + numel].view_as(p).clone()
            offset += numel

    @staticmethod
    def _capture_training_modes(module: nn.Module) -> dict[nn.Module, bool]:
        """Capture train/eval flags for a module and all descendants."""
        return {submodule: submodule.training for submodule in module.modules()}

    @staticmethod
    def _restore_training_modes(training_modes: dict[nn.Module, bool]) -> None:
        """Restore train/eval flags captured by ``_capture_training_modes``."""
        for module, is_training in training_modes.items():
            module.train(is_training)

    @torch.no_grad()
    def _compute_Q(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None,
        X_A_mask: torch.Tensor | None,
        Y_N: torch.Tensor,
        Y_N_mask: torch.Tensor,
        pred_mask: torch.Tensor | None = None,
        pred_denom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Q(θ) = prediction loss for the current imputer parameters.

        Runs under ``torch.no_grad()`` — no computational graph is built.
        Used by the zero-order estimator which temporarily perturbs
        parameters.
        """
        imputer_output = self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)
        imputed_X = imputer_output["imp_data"]
        predictor_output = self.predictor(imputed_X)
        pred_data = predictor_output.get("pred_data", None)

        if pred_data is None:
            return torch.tensor(0.0, device=self.device)

        if pred_mask is None:
            pred_mask = 1.0 - Y_N_mask
        if pred_denom is None:
            pred_denom = pred_mask.sum() + 1e-8

        diff = pred_data - Y_N
        return (diff.square() * pred_mask).sum() / pred_denom

    @torch.no_grad()
    def _compute_gp_hat(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None,
        X_A_mask: torch.Tensor | None,
        Y_N: torch.Tensor,
        Y_N_mask: torch.Tensor,
        k: int,
        epsilon: float,
        seed_parts: tuple[object, ...] = (),
    ) -> torch.Tensor:
        """Zero-order gradient estimation via symmetric differences.

        Computes::

            ĝ_P = (1/k) Σᵢ [Q(θ+ε·uᵢ) − Q(θ−ε·uᵢ)] / (2ε) · uᵢ
            where uᵢ ~ N(0, I).

        Returns ĝ_P as a flat 1-D tensor.  Always restores the original
        imputer parameters and train/eval mode before returning.
        """
        params = list(self.imputer.parameters())
        theta = [p.detach().clone() for p in params]
        gp_parts = [torch.zeros_like(p) for p in params]
        directions = [torch.empty_like(p) for p in params]
        pred_mask = 1.0 - Y_N_mask
        pred_denom = pred_mask.sum() + 1e-8
        training_modes = self._capture_training_modes(self.imputer)
        try:
            self.imputer.eval()

            for _ in range(k):
                for u in directions:
                    u.normal_(generator=self.tip_generator)

                # Q(θ + ε·u)
                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=epsilon)
                Q_plus = self._compute_Q(
                    X_NA, X_NA_mask, X_N, X_A_mask,
                    Y_N, Y_N_mask,
                    pred_mask=pred_mask, pred_denom=pred_denom,
                )

                # Q(θ − ε·u)
                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=-2.0 * epsilon)
                Q_minus = self._compute_Q(
                    X_NA, X_NA_mask, X_N, X_A_mask,
                    Y_N, Y_N_mask,
                    pred_mask=pred_mask, pred_denom=pred_denom,
                )

                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=epsilon)

                coeff = (Q_plus - Q_minus) / (2.0 * epsilon)
                for gp_part, u in zip(gp_parts, directions, strict=True):
                    gp_part.add_(u.mul_(coeff))

        finally:
            for p, base in zip(params, theta, strict=True):
                p.copy_(base)
            self._restore_training_modes(training_modes)

        return torch.cat([gp_part.flatten() for gp_part in gp_parts]).div_(k)

    @staticmethod
    def _apply_projection(
        g_I: torch.Tensor,
        gp_hat: torch.Tensor,
        conditional: bool,
    ) -> torch.Tensor:
        """Orthogonal projection: remove the component of *g_I* parallel to
        *gp_hat*.

        Args:
            g_I: Imputation gradient (flat 1-D).
            gp_hat: Zero-order prediction gradient estimate (flat 1-D).
            conditional: If ``True``, only project when
                ``g_Iᵀ·gp_hat < 0`` (conflict detected).  If ``False``,
                always project.

        Returns:
            (Potentially modified) gradient vector.
        """
        dot = torch.dot(g_I, gp_hat)

        should_project = (not conditional) or (dot < 0)

        if should_project:
            gp_norm_sq = torch.dot(gp_hat, gp_hat)
            if gp_norm_sq > 1e-12:
                projected = g_I - (dot / gp_norm_sq) * gp_hat
                return projected

        return g_I

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        data_loader: dict[str, DataLoader],
        phase_name: str,
        num_epoch: int,
        accum_steps: int | None = None,
        on_epoch_end: collections.abc.Callable[
            [int, float, JointModel], None
        ]
        | None = None,
        grad_modifier: collections.abc.Callable[
            [JointModel, list[dict[str, torch.Tensor]]], None
        ]
        | None = None,
    ) -> float:
        """Train the imputer for a fixed number of epochs.

        The predictor is never trained — it runs in ``eval()`` /
        ``no_grad()`` throughout.

        Args:
            data_loader: Dictionary with a ``"train"`` DataLoader.
            phase_name: Human-readable phase name (for logging).
            num_epoch: Number of epochs for this phase.
            accum_steps: Micro-batches to accumulate before one
                optimizer step.  Falls back to ``self.accum_steps``,
                then 1 (no accumulation).
            on_epoch_end: Optional callback ``(epoch, train_loss, model)``
                invoked after each training epoch.
            grad_modifier: Optional callback ``(model, microbatch_dicts)``
                invoked after gradient accumulation and before gradient
                clipping and optimizer step.  Used by TIP to read /
                modify imputer gradients.

        Returns:
            Final training loss from the last epoch in this phase.
        """
        if accum_steps is None:
            accum_steps = getattr(self, "accum_steps", 1)

        self._seed_training_stochastic_ops()

        train_loader = data_loader["train"]
        final_train_loss = float("inf")
        self.phase_epoch = 0

        while self.phase_epoch < num_epoch:
            total_loss = 0.0
            num_batches = len(train_loader)
            num_optim_steps = 0

            self.imputer.train()

            accum_loss = 0.0
            accum_count = 0
            accum_target = 1
            accum_microbatches: list[dict[str, torch.Tensor]] = []

            for batch_idx, batch in enumerate(train_loader):
                if accum_count == 0:
                    accum_target = min(accum_steps, num_batches - batch_idx)
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                    self._move_batch_to_device(batch)
                )

                output = self.forward(
                    X_NA,
                    X_NA_mask,
                    X_N,
                    X_A_mask,
                    Y_N,
                    Y_N_mask,
                    run_predictor=False,
                )

                loss = output["loss"]  # imp_loss only
                # Normalise by this accumulation window so each optimizer step
                # uses the mean gradient of the microbatches it actually saw.
                scaled_loss = loss / accum_target
                scaled_loss.backward()

                accum_loss += loss.item()
                accum_count += 1
                accum_microbatches.append(
                    {
                        "X_N": X_N,
                        "X_NA": X_NA,
                        "X_NA_mask": X_NA_mask,
                        "X_A_mask": X_A_mask,
                        "Y_N": Y_N,
                        "Y_N_mask": Y_N_mask,
                    }
                )

                is_last_batch = (batch_idx + 1) == num_batches
                if accum_count >= accum_target or is_last_batch:
                    # TIP grad_modifier hook — read / modify gradients
                    # after gradient accumulation and before clip + step.
                    if grad_modifier is not None:
                        grad_modifier(self, accum_microbatches)

                    torch.nn.utils.clip_grad_norm_(
                        self.imputer.parameters(),
                        max_norm=10.0,
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_optim_step += 1
                    total_loss += accum_loss / accum_count
                    num_optim_steps += 1

                    accum_loss = 0.0
                    accum_count = 0
                    accum_target = 1
                    accum_microbatches = []

            avg_train_loss = total_loss / max(1, num_optim_steps)
            final_train_loss = avg_train_loss

            self.logger.info(
                "Epoch %03d/%d | imp_loss=%.6f",
                self.phase_epoch + 1,
                num_epoch,
                avg_train_loss,
            )

            if on_epoch_end is not None:
                on_epoch_end(
                    self.global_epoch + 1,
                    avg_train_loss,
                    self,
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.phase_epoch += 1
            self.global_epoch += 1

        self.logger.info(
            f"Finished Phase: {phase_name}. Final Train Loss: {final_train_loss:.6f}"
        )

        return final_train_loss

    def fit_tip(
        self,
        data_loader: dict[str, DataLoader],
        phase_name: str,
        num_epoch: int,
        variant: str = "B2",
        k: int = 20,
        epsilon: float = 1e-4,
        accum_steps: int | None = None,
        on_epoch_end: collections.abc.Callable[
            [int, float, JointModel], None
        ]
        | None = None,
    ) -> float:
        """TIP training via a gradient-modifier callback.

        Delegates to :meth:`fit` with a closure that implements:
        - ``"B2"``: zero-order ĝ_P estimation + conditional projection.
        - ``"B4"``: random direction replaces ĝ_P + conditional projection.
        - ``"B5"``: zero-order ĝ_P estimation + unconditional (always)
          projection.

        Args:
            data_loader: Dictionary with a ``"train"`` DataLoader.
            phase_name: Human-readable phase name (for logging).
            num_epoch: Number of epochs for this phase.
            variant: One of ``"B2"``, ``"B4"``, ``"B5"``, ``"B6"``.
            k: Number of random samples for zero-order estimation
                (ignored by B4).
            epsilon: Perturbation step size (ignored by B4).
            accum_steps: Micro-batches per optimizer step.
            on_epoch_end: Optional per-epoch callback.

        Returns:
            Final training loss from the last epoch in this phase.
        """
        if variant not in {"B2", "B4", "B5", "B6"}:
            raise ValueError(f"Unknown TIP variant: {variant}")

        tip_stats = {
            "steps": 0,
            "conflicts": 0,
            "projections": 0,
            "dot_sum": 0.0,
            "gp_norm_sum": 0.0,
        }

        def _tip_modifier(
            model: JointModel, microbatches: list[dict[str, torch.Tensor]]
        ) -> None:
            g_I = model._get_flattened_grad()

            if variant == "B4":
                # Random direction — replaces gp_hat entirely
                gp_hat = torch.randn(
                    g_I.shape,
                    dtype=g_I.dtype,
                    device=g_I.device,
                    generator=model.tip_generator,
                )
            else:
                # B2 / B5: average zero-order prediction gradients over
                # the same microbatches that produced the accumulated
                # imputation gradient.
                gp_hat = torch.zeros_like(g_I)
                for microbatch_idx, batch_data in enumerate(microbatches):
                    gp_hat.add_(
                        model._compute_gp_hat(
                            batch_data["X_NA"],
                            batch_data["X_NA_mask"],
                            batch_data["X_N"],
                            batch_data["X_A_mask"],
                            batch_data["Y_N"],
                            batch_data["Y_N_mask"],
                            k=k,
                            epsilon=epsilon,
                            seed_parts=("microbatch", microbatch_idx),
                        )
                    )
                gp_hat.div_(max(1, len(microbatches)))

            dot = torch.dot(g_I, gp_hat).detach()
            gp_norm = torch.linalg.vector_norm(gp_hat).detach()
            conflict = dot.item() < 0.0
            projection = (
                variant != "B6"
                and (variant == "B5" or conflict)
                and gp_norm.item() > 1e-12
            )

            tip_stats["steps"] += 1
            tip_stats["conflicts"] += int(conflict)
            tip_stats["projections"] += int(projection)
            tip_stats["dot_sum"] += float(dot.item())
            tip_stats["gp_norm_sum"] += float(gp_norm.item())

            if variant == "B6":
                model._set_flattened_grad(gp_hat)
                return

            conditional = variant != "B5"
            g_I = model._apply_projection(
                g_I,
                gp_hat,
                conditional=conditional,
            )
            model._set_flattened_grad(g_I)

        def _tip_epoch_end(
            epoch: int,
            train_loss: float,
            model: JointModel,
        ) -> None:
            steps = int(tip_stats["steps"])
            if steps > 0:
                conflict_rate = tip_stats["conflicts"] / steps
                model.logger.info(
                    "TIP conflict_rate=%.4f | steps=%d",
                    conflict_rate,
                    steps,
                )
                model.tb_summary_writer.add_scalar(
                    f"tip/{phase_name}/conflict_rate",
                    conflict_rate,
                    epoch,
                )

            for key in tip_stats:
                tip_stats[key] = 0

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, model)

        return self.fit(
            data_loader,
            phase_name,
            num_epoch,
            accum_steps=accum_steps,
            on_epoch_end=_tip_epoch_end,
            grad_modifier=_tip_modifier,
        )
