"""Joint training wrapper for imputation and prediction models.

The imputer is trained via its reconstruction / imputation loss.  The
predictor is **inference-only** — it runs inside ``torch.no_grad()``,
does not produce a loss, and never participates in backpropagation.
"""

from __future__ import annotations

import collections.abc
import logging
import math

from scipy.stats import t as student_t
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

    @torch.no_grad()
    def compute_val_loss(self, val_loader: DataLoader) -> float:
        """Average imputation loss on the validation set.

        Losses are weighted by batch size so the last (potentially
        smaller) batch does not distort the validation metric used for
        early stopping and checkpoint selection.
        """
        if val_loader is None:
            return float("inf")
        self.imputer.eval()
        total = 0.0
        total_samples = 0
        no_gt = getattr(val_loader, "no_gt", False)
        for batch in val_loader:
            tensors = self._move_batch_to_device(batch)
            if len(tensors) == 6:
                X_N, X_NA, X_NA_mask, X_A_mask, _Y_N, _Y_N_mask = tensors
            elif no_gt:
                raise RuntimeError(
                    "Validation imputation loss requires artificial masks; "
                    "use the training-format validation loader."
                )
            else:
                X_NA, X_NA_mask, X_N, _ = tensors
                X_A_mask = X_NA_mask
            bs = X_NA.size(0)
            total += self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)["imp_loss"].item() * bs
            total_samples += bs
        self.imputer.train()
        return total / max(total_samples, 1)

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
        alignment_grads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Estimate the prediction gradient and its directional alignments.

        TSFM forward calls are batched: all *2k* imputed outputs are
        stacked and fed to the predictor in a single call, reducing the
        dominant cost from :math:`O(k)` to :math:`O(1)` TSFM forwards.
        """
        params = list(self.imputer.parameters())
        theta = [p.detach().clone() for p in params]
        gp_parts = [torch.zeros_like(p) for p in params]
        directions = [torch.empty_like(p) for p in params]
        pred_mask = 1.0 - Y_N_mask
        pred_denom = pred_mask.sum() + 1e-8

        # ---- phase 1: perturb, run imputer forward, collect outputs ----
        saved_dirs: list[list[torch.Tensor]] = []
        imputed_batch: list[torch.Tensor] = []
        training_modes = self._capture_training_modes(self.imputer)
        try:
            self.imputer.eval()
            for _ in range(k):
                for u in directions:
                    u.normal_(generator=self.tip_generator)
                saved_dirs.append([u.clone() for u in directions])

                # +ε  perturbation
                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=epsilon)
                imputed_batch.append(
                    self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )

                # −ε  perturbation  (from +ε to −ε in one step)
                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=-2.0 * epsilon)
                imputed_batch.append(
                    self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )

                # restore to θ
                for p, u in zip(params, directions, strict=True):
                    p.add_(u, alpha=epsilon)

            # ---- phase 2: single batched TSFM forward ----
            batched_input = torch.cat(imputed_batch, dim=0)          # [2k·B, L, D]
            batched_pred = self.predictor(batched_input)              # single call
            pred_data = batched_pred.get("pred_data", None)
            if pred_data is None:
                return (
                    torch.cat([gp_part.flatten() for gp_part in gp_parts]).div_(k),
                    torch.zeros(k, device=self.device),
                    {},
                )

            # ---- phase 3: slice predictions, compute losses, accumulate ----
            B = X_NA.shape[0]
            z_samples: list[torch.Tensor] = []
            q_plus_samples: list[torch.Tensor] = []
            q_minus_samples: list[torch.Tensor] = []
            coeff_samples: list[torch.Tensor] = []
            feature_coeff_samples: list[torch.Tensor] = []
            horizon_coeff_samples: list[torch.Tensor] = []
            feature_denom = pred_mask.sum(dim=(0, 1))
            horizon_denom = pred_mask.sum(dim=(0, 2))
            for i in range(k):
                p_plus = pred_data[2 * i * B : (2 * i + 1) * B]
                p_minus = pred_data[(2 * i + 1) * B : (2 * i + 2) * B]

                Q_plus = ((p_plus - Y_N).square() * pred_mask).sum() / pred_denom
                Q_minus = ((p_minus - Y_N).square() * pred_mask).sum() / pred_denom

                coeff = (Q_plus - Q_minus) / (2.0 * epsilon)
                q_plus_samples.append(Q_plus)
                q_minus_samples.append(Q_minus)
                coeff_samples.append(coeff)

                feature_q_plus = (
                    (p_plus - Y_N).square() * pred_mask
                ).sum(dim=(0, 1)) / feature_denom.clamp_min(1.0)
                feature_q_minus = (
                    (p_minus - Y_N).square() * pred_mask
                ).sum(dim=(0, 1)) / feature_denom.clamp_min(1.0)
                feature_coeff_samples.append(
                    (feature_q_plus - feature_q_minus) / (2.0 * epsilon)
                )

                horizon_q_plus = (
                    (p_plus - Y_N).square() * pred_mask
                ).sum(dim=(0, 2)) / horizon_denom.clamp_min(1.0)
                horizon_q_minus = (
                    (p_minus - Y_N).square() * pred_mask
                ).sum(dim=(0, 2)) / horizon_denom.clamp_min(1.0)
                horizon_coeff_samples.append(
                    (horizon_q_plus - horizon_q_minus) / (2.0 * epsilon)
                )
                dirs_i = saved_dirs[i]
                alignment = torch.zeros((), device=self.device, dtype=coeff.dtype)
                for g_i, u in zip(alignment_grads, dirs_i, strict=True):
                    alignment.add_(torch.sum(g_i * u) * coeff)
                z_samples.append(alignment)
                for gp_part, u in zip(gp_parts, dirs_i, strict=True):
                    gp_part.add_(u * coeff)
        finally:
            for p, base in zip(params, theta, strict=True):
                p.copy_(base)
            self._restore_training_modes(training_modes)

        coeff_tensor = torch.stack(coeff_samples)
        feature_coeff = torch.stack(feature_coeff_samples)
        horizon_coeff = torch.stack(horizon_coeff_samples)

        def cancellation_ratio(
            component_coeff: torch.Tensor, component_weight: torch.Tensor
        ) -> torch.Tensor:
            weights = component_weight / component_weight.sum().clamp_min(1.0)
            weighted_abs = (component_coeff.abs() * weights).sum(dim=1)
            ratio = (
                1.0 - coeff_tensor.abs() / weighted_abs.clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            return torch.where(weighted_abs > 1e-12, ratio, torch.zeros_like(ratio))

        diagnostics = {
            "q_plus": torch.stack(q_plus_samples),
            "q_minus": torch.stack(q_minus_samples),
            "coeff": coeff_tensor,
            "feature_coeff": feature_coeff,
            "horizon_coeff": horizon_coeff,
            "feature_cancellation": cancellation_ratio(feature_coeff, feature_denom),
            "horizon_cancellation": cancellation_ratio(horizon_coeff, horizon_denom),
        }
        return (
            torch.cat([gp_part.flatten() for gp_part in gp_parts]).div_(k),
            torch.stack(z_samples),
            diagnostics,
        )

    @staticmethod
    def _student_t_upper_confidence_bound(
        samples: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        """Return the one-sided Student-t upper confidence bound for a mean."""
        if samples.numel() < 2:
            return torch.full((), float("inf"), device=samples.device)
        if not 0.0 < delta < 1.0:
            raise ValueError(f"confidence delta must lie in (0, 1), got {delta}")

        critical = float(student_t.ppf(1.0 - delta, df=samples.numel() - 1))
        if not math.isfinite(critical):
            return torch.full((), float("inf"), device=samples.device)
        standard_error = samples.std(unbiased=True) / math.sqrt(samples.numel())
        return samples.mean() + critical * standard_error

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

            if on_epoch_end is not None:
                should_stop = on_epoch_end(
                    self.global_epoch + 1,
                    avg_train_loss,
                    self,
                )
                if should_stop:
                    break

            self.phase_epoch += 1
            self.global_epoch += 1

        self.logger.info(
            f"Finished Phase: {phase_name}. Final Train Loss: {final_train_loss:.6f}"
        )

        return final_train_loss

    def _fit_tip_sign_gate_legacy(
        self,
        data_loader: dict[str, DataLoader],
        phase_name: str,
        num_epoch: int,
        k: int = 20,
        epsilon: float = 1e-4,
        accum_steps: int | None = None,
        on_epoch_end: collections.abc.Callable[
            [int, float, JointModel], None
        ]
        | None = None,
    ) -> float:
        """TIP training via zero-order ĝ_P estimation + conditional projection.

        Delegates to :meth:`fit` with a gradient-modifier closure that
        estimates the prediction gradient via symmetric differences and
        applies conditional orthogonal projection.

        Args:
            data_loader: Dictionary with a ``"train"`` DataLoader.
            phase_name: Human-readable phase name (for logging).
            num_epoch: Number of epochs for this phase.
            k: Number of random samples for zero-order estimation.
            epsilon: Perturbation step size.
            accum_steps: Micro-batches per optimizer step.
            on_epoch_end: Optional per-epoch callback.

        Returns:
            Final training loss from the last epoch in this phase.
        """
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

            # Average zero-order prediction gradients over the same
            # microbatches that produced the accumulated imputation gradient.
            gp_hat = torch.zeros_like(g_I)
            for _microbatch_idx, batch_data in enumerate(microbatches):
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
                        alignment_grads=[
                            p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                            for p in model.imputer.parameters()
                        ],
                    )[0]
                )
            gp_hat.div_(max(1, len(microbatches)))

            dot = torch.dot(g_I, gp_hat).detach()
            gp_norm = torch.linalg.vector_norm(gp_hat).detach()
            conflict = dot.item() < 0.0
            projection = conflict and gp_norm.item() > 1e-12

            tip_stats["steps"] += 1
            tip_stats["conflicts"] += int(conflict)
            tip_stats["projections"] += int(projection)
            tip_stats["dot_sum"] += float(dot.item())
            tip_stats["gp_norm_sum"] += float(gp_norm.item())

            g_I = model._apply_projection(g_I, gp_hat, conditional=True)
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
                return on_epoch_end(epoch, train_loss, model)
            return False

        return self.fit(
            data_loader,
            phase_name,
            num_epoch,
            accum_steps=accum_steps,
            on_epoch_end=_tip_epoch_end,
            grad_modifier=_tip_modifier,
        )


    def fit_tip(
        self, data_loader, phase_name, num_epoch, k=20, epsilon=1e-4,
        confidence_delta=0.05, accum_steps=None, on_epoch_end=None,
    ):
        """TIP with Student-t confidence-gated projection."""
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must lie in (0, 1)")
        if accum_steps is None:
            accum_steps = self.accum_steps
        self.logger.info(
            "TIP confidence gate: step_delta=%g",
            confidence_delta,
        )
        scalar_keys = (
            "ucb", "gate_margin", "z_mean", "z_std", "z_sem",
            "z_negative_fraction", "alignment_estimation_gap",
            "q_center", "q_pair_gap_abs", "coeff_abs", "coeff_rms",
            "gi_norm", "gp_norm", "dot", "cosine",
            "candidate_correction_ratio", "candidate_retained_ratio",
            "candidate_post_dot_abs", "applied_correction_ratio",
            "feature_cancellation", "horizon_cancellation",
        )
        stats = {
            "steps": 0,
            "raw": 0,
            "projected": 0,
            "uncertain_conflict": 0,
            **{key: 0.0 for key in scalar_keys},
            "feature_abs": None,
            "feature_signed": None,
            "horizon_abs": None,
            "horizon_signed": None,
        }

        def modifier(model, microbatches):
            g_i = model._get_flattened_grad()
            grad_parts = [
                p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                for p in model.imputer.parameters()
            ]
            gp_hat = torch.zeros_like(g_i)
            all_z = []
            all_diagnostics: list[dict[str, torch.Tensor]] = []
            for batch in microbatches:
                batch_gp, z, diagnostics = model._compute_gp_hat(
                    batch["X_NA"], batch["X_NA_mask"], batch["X_N"],
                    batch["X_A_mask"], batch["Y_N"], batch["Y_N_mask"],
                    k=k, epsilon=epsilon, alignment_grads=grad_parts,
                )
                gp_hat.add_(batch_gp)
                all_z.append(z)
                if diagnostics:
                    all_diagnostics.append(diagnostics)
            gp_hat.div_(len(microbatches))
            z_samples = torch.cat(all_z)
            upper = model._student_t_upper_confidence_bound(
                z_samples, confidence_delta
            )
            dot = torch.dot(g_i, gp_hat)
            gi_norm = torch.linalg.vector_norm(g_i)
            gp_norm = torch.linalg.vector_norm(gp_hat)
            cosine = dot / (gi_norm * gp_norm).clamp_min(1e-12)
            raw = dot.item() < 0.0
            project = upper.item() < 0.0 and gp_norm.item() > 1e-12

            candidate = model._apply_projection(g_i, gp_hat, conditional=False)
            correction = g_i - candidate
            correction_ratio = (
                torch.linalg.vector_norm(correction) / gi_norm.clamp_min(1e-12)
            )
            retained_ratio = (
                torch.linalg.vector_norm(candidate) / gi_norm.clamp_min(1e-12)
            )
            candidate_post_dot_abs = torch.dot(candidate, gp_hat).abs()

            stats["steps"] += 1
            stats["raw"] += int(raw)
            stats["projected"] += int(project)
            stats["uncertain_conflict"] += int(raw and not project)
            stats["ucb"] += upper.item()
            z_mean = z_samples.mean()
            z_std = z_samples.std(unbiased=z_samples.numel() > 1)
            stats["gate_margin"] += -upper.item()
            stats["z_mean"] += z_mean.item()
            stats["z_std"] += z_std.item()
            stats["z_sem"] += (z_std / math.sqrt(z_samples.numel())).item()
            stats["z_negative_fraction"] += (z_samples < 0).float().mean().item()
            stats["alignment_estimation_gap"] += (dot - z_mean).item()
            stats["gi_norm"] += gi_norm.item()
            stats["gp_norm"] += gp_norm.item()
            stats["dot"] += dot.item()
            stats["cosine"] += cosine.item()
            stats["candidate_correction_ratio"] += correction_ratio.item()
            stats["candidate_retained_ratio"] += retained_ratio.item()
            stats["candidate_post_dot_abs"] += candidate_post_dot_abs.item()
            stats["applied_correction_ratio"] += (
                correction_ratio.item() if project else 0.0
            )

            if all_diagnostics:
                q_plus = torch.cat([d["q_plus"] for d in all_diagnostics])
                q_minus = torch.cat([d["q_minus"] for d in all_diagnostics])
                coeff = torch.cat([d["coeff"] for d in all_diagnostics])
                feature_coeff = torch.cat(
                    [d["feature_coeff"] for d in all_diagnostics]
                )
                horizon_coeff = torch.cat(
                    [d["horizon_coeff"] for d in all_diagnostics]
                )
                stats["q_center"] += ((q_plus + q_minus) * 0.5).mean().item()
                stats["q_pair_gap_abs"] += (q_plus - q_minus).abs().mean().item()
                stats["coeff_abs"] += coeff.abs().mean().item()
                stats["coeff_rms"] += coeff.square().mean().sqrt().item()
                stats["feature_cancellation"] += torch.cat(
                    [d["feature_cancellation"] for d in all_diagnostics]
                ).mean().item()
                stats["horizon_cancellation"] += torch.cat(
                    [d["horizon_cancellation"] for d in all_diagnostics]
                ).mean().item()

                for key, value in (
                    ("feature_abs", feature_coeff.abs().mean(dim=0)),
                    ("feature_signed", feature_coeff.mean(dim=0)),
                    ("horizon_abs", horizon_coeff.abs().mean(dim=0)),
                    ("horizon_signed", horizon_coeff.mean(dim=0)),
                ):
                    value = value.detach().float().cpu()
                    stats[key] = value if stats[key] is None else stats[key] + value
            if project:
                model._set_flattened_grad(
                    candidate
                )

        def epoch_end(epoch, train_loss, model):
            if stats["steps"]:
                raw_rate = stats["raw"] / stats["steps"]
                projection_rate = stats["projected"] / stats["steps"]
                mean_ucb = stats["ucb"] / stats["steps"]
                uncertain_rate = stats["uncertain_conflict"] / stats["steps"]
                model.logger.info(
                    "TIP raw_conflict_rate=%.4f | projection_rate=%.4f | "
                    "mean_ucb=%.6g | steps=%d | uncertain_conflict_rate=%.4f",
                    raw_rate, projection_rate, mean_ucb, stats["steps"],
                    uncertain_rate,
                )
                averages = {
                    key: stats[key] / stats["steps"] for key in scalar_keys
                }
                model.logger.info(
                    "TIP estimator: z_mean=%.6g | z_std=%.6g | z_sem=%.6g | "
                    "z_negative_fraction=%.4f | coeff_abs=%.6g | "
                    "coeff_rms=%.6g | q_center=%.6g | q_pair_gap_abs=%.6g | "
                    "gate_margin=%.6g | alignment_gap=%.6g",
                    averages["z_mean"], averages["z_std"], averages["z_sem"],
                    averages["z_negative_fraction"], averages["coeff_abs"],
                    averages["coeff_rms"], averages["q_center"],
                    averages["q_pair_gap_abs"], averages["gate_margin"],
                    averages["alignment_estimation_gap"],
                )
                model.logger.info(
                    "TIP geometry: gi_norm=%.6g | gp_norm=%.6g | dot=%.6g | "
                    "cosine=%.6g | candidate_correction_ratio=%.6g | "
                    "candidate_retained_ratio=%.6g | applied_correction_ratio=%.6g | "
                    "post_dot_abs=%.6g",
                    averages["gi_norm"], averages["gp_norm"], averages["dot"],
                    averages["cosine"], averages["candidate_correction_ratio"],
                    averages["candidate_retained_ratio"],
                    averages["applied_correction_ratio"],
                    averages["candidate_post_dot_abs"],
                )
                model.logger.info(
                    "TIP decomposition: feature_cancellation=%.4f | "
                    "horizon_cancellation=%.4f",
                    averages["feature_cancellation"],
                    averages["horizon_cancellation"],
                )

                tb_values = {
                    "raw_conflict_rate": raw_rate,
                    "projection_rate": projection_rate,
                    "uncertain_conflict_rate": uncertain_rate,
                    "mean_ucb": mean_ucb,
                    **averages,
                }
                for name, value in tb_values.items():
                    model.tb_summary_writer.add_scalar(
                        f"tip/{phase_name}/{name}", value, epoch
                    )

                for axis in ("feature", "horizon"):
                    abs_values = stats[f"{axis}_abs"]
                    signed_values = stats[f"{axis}_signed"]
                    if abs_values is None or signed_values is None:
                        continue
                    abs_values = abs_values / stats["steps"]
                    signed_values = signed_values / stats["steps"]
                    for idx in range(min(64, abs_values.numel())):
                        model.tb_summary_writer.add_scalar(
                            f"tip/{phase_name}/{axis}_{idx:03d}_abs_coeff",
                            abs_values[idx].item(), epoch,
                        )
                        model.tb_summary_writer.add_scalar(
                            f"tip/{phase_name}/{axis}_{idx:03d}_signed_coeff",
                            signed_values[idx].item(), epoch,
                        )
                    top_count = min(5, abs_values.numel())
                    top = torch.topk(abs_values, k=top_count)
                    top_summary = ", ".join(
                        f"{int(idx)}:{float(value):.4g}"
                        for value, idx in zip(top.values, top.indices, strict=True)
                    )
                    model.logger.info(
                        "TIP %s sensitivity top-%d (index:mean_abs_coeff): %s",
                        axis, top_count, top_summary,
                    )
            for key in stats:
                stats[key] = None if key in {
                    "feature_abs", "feature_signed", "horizon_abs", "horizon_signed"
                } else (0.0 if key in scalar_keys else 0)
            return on_epoch_end(epoch, train_loss, model) if on_epoch_end else False

        return self.fit(
            data_loader, phase_name, num_epoch, accum_steps=accum_steps,
            on_epoch_end=epoch_end, grad_modifier=modifier,
        )
