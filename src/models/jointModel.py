"""Joint training wrapper for imputation and prediction models.

The imputer is trained via its reconstruction / imputation loss.  The
predictor is **inference-only** — it runs inside ``torch.no_grad()``,
does not produce a loss, and never participates in backpropagation.
"""

from __future__ import annotations

import collections.abc
import logging
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.rng import derive_seed, torch_seed_scope

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
        self.logger.info(
            "Configured independent RNG: training_stochastic_seed=%d",
            self.training_stochastic_seed,
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

    @staticmethod
    def _orthonormal_update_basis(
        vectors: list[torch.Tensor],
        max_rank: int,
        relative_tolerance: float = 1e-6,
    ) -> torch.Tensor:
        """Build a local orthonormal basis with the current update first."""
        basis: list[torch.Tensor] = []
        for vector in vectors:
            residual = vector.detach().clone()
            original_norm = torch.linalg.vector_norm(residual)
            if original_norm <= 1e-12:
                continue
            for _ in range(2):
                for axis in basis:
                    residual.sub_(axis, alpha=torch.dot(axis, residual))
            residual_norm = torch.linalg.vector_norm(residual)
            if residual_norm / original_norm > relative_tolerance:
                basis.append(residual / residual_norm)
            if len(basis) >= max_rank:
                break
        if not basis:
            raise RuntimeError("Cannot construct a TIP subspace from zero updates")
        return torch.stack(basis, dim=1)

    @torch.no_grad()
    def _compute_subspace_prediction_gradient(
        self,
        X_NA: torch.Tensor,
        X_NA_mask: torch.Tensor,
        X_N: torch.Tensor | None,
        X_A_mask: torch.Tensor | None,
        Y_N: torch.Tensor,
        Y_N_mask: torch.Tensor,
        epsilon: float,
        basis: torch.Tensor,
        update_coordinates: torch.Tensor,
        rng_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Measure every coordinate of the local prediction gradient.

        The ``2r`` central-difference queries use two batched predictor calls:
        one for all positive perturbations and one for all negative
        perturbations. Both calls share common random numbers. Confidence
        samples are paired per-window derivatives along the actual update.
        """
        params = list(self.imputer.parameters())
        theta = [p.detach().clone() for p in params]
        pred_mask = 1.0 - Y_N_mask
        rank = basis.shape[1]

        imputed_plus: list[torch.Tensor] = []
        imputed_minus: list[torch.Tensor] = []
        training_modes = self._capture_training_modes(self.imputer)

        def add_flat_direction(direction: torch.Tensor, alpha: float) -> None:
            offset = 0
            for parameter in params:
                count = parameter.numel()
                parameter.add_(
                    direction[offset : offset + count].view_as(parameter),
                    alpha=alpha,
                )
                offset += count

        try:
            self.imputer.eval()
            for coordinate in range(rank):
                direction = basis[:, coordinate]
                add_flat_direction(direction, epsilon)
                imputed_plus.append(
                    self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )
                add_flat_direction(direction, -2.0 * epsilon)
                imputed_minus.append(
                    self.imputer(X_NA, X_NA_mask, X_N, X_A_mask)["imp_data"]
                )
                add_flat_direction(direction, epsilon)

            # Common random numbers: Sundial samples fresh flow noise in every
            # generate() call. Resetting the same scoped seed for the equally
            # shaped +epsilon and -epsilon batches makes corresponding rows
            # use identical noise, while fork_rng restores the training RNG.
            prediction_seed_parts = (
                "tip_finite_difference",
                self.global_optim_step,
                rng_index,
            )
            with torch_seed_scope(self.base_seed, *prediction_seed_parts):
                plus_output = self.predictor(torch.cat(imputed_plus, dim=0))
            with torch_seed_scope(self.base_seed, *prediction_seed_parts):
                minus_output = self.predictor(torch.cat(imputed_minus, dim=0))
            plus_data = plus_output.get("pred_data", None)
            minus_data = minus_output.get("pred_data", None)
            if plus_data is None or minus_data is None:
                return (
                    torch.zeros(rank, device=self.device),
                    torch.zeros(X_NA.shape[0], device=self.device),
                    {},
                )

            batch_size = X_NA.shape[0]
            sample_denom = pred_mask.sum(dim=(1, 2)).clamp_min(1.0)
            coordinate_samples: list[torch.Tensor] = []
            q_plus_samples: list[torch.Tensor] = []
            q_minus_samples: list[torch.Tensor] = []
            for coordinate in range(rank):
                plus = plus_data[
                    coordinate * batch_size : (coordinate + 1) * batch_size
                ]
                minus = minus_data[
                    coordinate * batch_size : (coordinate + 1) * batch_size
                ]
                q_plus = ((plus - Y_N).square() * pred_mask).sum(
                    dim=(1, 2)
                ) / sample_denom
                q_minus = ((minus - Y_N).square() * pred_mask).sum(
                    dim=(1, 2)
                ) / sample_denom
                q_plus_samples.append(q_plus)
                q_minus_samples.append(q_minus)
                coordinate_samples.append((q_plus - q_minus) / (2.0 * epsilon))
        finally:
            for p, base in zip(params, theta, strict=True):
                p.copy_(base)
            self._restore_training_modes(training_modes)

        # [B, r]: each row is one window's paired low-dimensional gradient.
        per_sample_coordinates = torch.stack(coordinate_samples, dim=1)
        gradient_coordinates = per_sample_coordinates.mean(dim=0)
        z_samples = per_sample_coordinates @ update_coordinates
        diagnostics = {
            "q_center": 0.5 * (
                torch.stack(q_plus_samples) + torch.stack(q_minus_samples)
            ),
            "coordinate_samples": per_sample_coordinates,
        }
        return gradient_coordinates, z_samples, diagnostics

    @torch.no_grad()
    def _prediction_loss_on_microbatches(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        rng_namespace: str | None = None,
    ) -> float | None:
        """Evaluate prediction MSE on one optimizer step's exact samples.

        The imputer is temporarily put in evaluation mode so the before/after
        comparison measures only the parameter update, not dropout noise.  All
        accumulated micro-batches are concatenated and use the same target
        mask in both evaluations.
        """
        if not microbatches:
            return None

        def concatenate(key: str) -> torch.Tensor | None:
            values = [batch[key] for batch in microbatches]
            if any(value is None for value in values):
                return None
            return torch.cat(values, dim=0)

        training_modes = self._capture_training_modes(self.imputer)
        try:
            self.imputer.eval()
            inputs = (
                concatenate("X_NA"),
                concatenate("X_NA_mask"),
                concatenate("X_N"),
                concatenate("X_A_mask"),
                concatenate("Y_N"),
                concatenate("Y_N_mask"),
            )
            if rng_namespace is None:
                output = self.forward(*inputs, run_predictor=True)
            else:
                with torch_seed_scope(
                    self.base_seed,
                    rng_namespace,
                    self.global_optim_step,
                ):
                    output = self.forward(*inputs, run_predictor=True)
            pred_loss = output["pred_loss"]
            return pred_loss.item() if pred_loss is not None else None
        finally:
            self._restore_training_modes(training_modes)

    def record_prediction_optimizer_step(
        self,
        *,
        phase_name: str,
        pred_loss_before: float,
        pred_loss_after_raw_adamw: float,
        pred_loss_after_applied_update: float,
        projection_applied: bool,
    ) -> None:
        """Persist one optimizer step's prediction-loss transition."""
        step = self.global_optim_step + 1
        denominator = max(abs(pred_loss_before), 1e-12)
        raw_delta = pred_loss_after_raw_adamw - pred_loss_before
        applied_delta = pred_loss_after_applied_update - pred_loss_before
        projection_gain = (
            pred_loss_after_raw_adamw - pred_loss_after_applied_update
        )
        values = {
            "pred_loss_before": pred_loss_before,
            "pred_loss_after": pred_loss_after_applied_update,
            "pred_loss_delta": applied_delta,
            "pred_loss_decrease": -applied_delta,
            "pred_loss_relative_delta": applied_delta / denominator,
            "raw_adamw_pred_loss_after": pred_loss_after_raw_adamw,
            "raw_adamw_pred_loss_delta": raw_delta,
            "raw_adamw_pred_loss_decrease": -raw_delta,
            "raw_adamw_pred_loss_relative_delta": raw_delta / denominator,
            "tip_projection_gain": projection_gain,
            "projection_applied": float(projection_applied),
        }
        for name, value in values.items():
            self.tb_summary_writer.add_scalar(
                f"prediction_step/{name}", value, step
            )
        self.logger.info(
            "Prediction optimizer step: step=%d | phase=%s | "
            "pred_loss_before=%.9g | raw_adamw_pred_loss_after=%.9g | "
            "pred_loss_after=%.9g | raw_adamw_pred_loss_delta=%.9g | "
            "pred_loss_delta=%.9g | pred_loss_relative_delta=%.9g | "
            "tip_projection_gain=%.9g | projection_applied=%d | "
            "pred_loss_decrease=%.9g | raw_adamw_pred_loss_decrease=%.9g",
            step,
            phase_name,
            pred_loss_before,
            pred_loss_after_raw_adamw,
            pred_loss_after_applied_update,
            raw_delta,
            applied_delta,
            applied_delta / denominator,
            projection_gain,
            int(projection_applied),
            -applied_delta,
            -raw_delta,
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
    def _robust_subspace_projection(
        update_coordinates: torch.Tensor,
        coordinate_samples: torch.Tensor,
        delta: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Project onto a confidence cone for the prediction direction.

        The returned coordinates minimise Euclidean deviation from the raw
        imputation update subject to

            mean(c)^T x - beta * sqrt(x^T Cov(mean(c)) x) >= 0.

        This accounts for uncertainty in the *whole* subspace direction,
        whereas the scalar UCB only decides whether the raw update conflicts.
        The problem is tiny (the configured TIP rank, normally four) and is
        solved on CPU.  A failed or statistically unidentified cone falls back
        to the zero coordinate vector, which safely skips the conflicting step.
        """
        if coordinate_samples.ndim != 2:
            raise ValueError("coordinate_samples must have shape [n, rank]")
        sample_count, rank = coordinate_samples.shape
        if sample_count < 2:
            return torch.zeros_like(update_coordinates), {
                "success": 0.0,
                "margin": 0.0,
                "fallback": 1.0,
            }

        mean = coordinate_samples.mean(dim=0).detach().double().cpu().numpy()
        samples = coordinate_samples.detach().double().cpu().numpy()
        raw = update_coordinates.detach().double().cpu().numpy()
        covariance_mean = np.cov(samples, rowvar=False, ddof=1) / sample_count
        covariance_mean = np.atleast_2d(covariance_mean)
        ridge = max(
            float(np.trace(covariance_mean)) / max(rank, 1) * 1e-8,
            1e-12,
        )
        covariance_mean = covariance_mean + ridge * np.eye(rank)
        beta = float(student_t.ppf(1.0 - delta, df=sample_count - 1))

        def objective(value: np.ndarray) -> float:
            residual = value - raw
            return 0.5 * float(residual @ residual)

        def robust_margin(value: np.ndarray) -> float:
            variance = max(float(value @ covariance_mean @ value), 0.0)
            return float(mean @ value - beta * math.sqrt(variance))

        starts = [np.zeros_like(raw), raw.copy()]
        mean_norm_sq = float(mean @ mean)
        if mean_norm_sq > 1e-20:
            starts.append(raw - float(raw @ mean) / mean_norm_sq * mean)
            starts.append(mean * (max(float(mean @ raw), 0.0) + 1.0) / mean_norm_sq)

        best = None
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                constraints={"type": "ineq", "fun": robust_margin},
                options={"ftol": 1e-12, "maxiter": 200},
            )
            if (
                result.success
                and robust_margin(result.x) >= -1e-7
                and (best is None or result.fun < best.fun)
            ):
                best = result

        if best is None:
            solution = np.zeros_like(raw)
            success = 0.0
            fallback = 1.0
        else:
            solution = best.x
            success = 1.0
            fallback = float(np.linalg.norm(solution) <= 1e-10)
        output = torch.as_tensor(
            solution,
            dtype=update_coordinates.dtype,
            device=update_coordinates.device,
        )
        return output, {
            "success": success,
            "margin": robust_margin(solution),
            "fallback": fallback,
        }

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
        optimizer_update_modifier: collections.abc.Callable[
            [JointModel, list[dict[str, torch.Tensor]], torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        optimizer_step_observer: collections.abc.Callable[
            [float, float, float], None
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
            optimizer_update_modifier: Optional callback receiving the model,
                accumulated micro-batches, and the actual AdamW update vector
                ``theta_before - theta_after``. It returns the update vector
                that will be applied to the parameters.
            optimizer_step_observer: Optional callback receiving prediction
                loss on the exact same samples before the update, after the
                raw AdamW proposal, and after the TIP-projected AdamW update.
                Enabling it adds three predictor forward calls per step.

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
                    torch.nn.utils.clip_grad_norm_(
                        self.imputer.parameters(),
                        max_norm=10.0,
                    )

                    if optimizer_update_modifier is not None:
                        parameters = list(self.imputer.parameters())
                        parameters_before = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_before = self._prediction_loss_on_microbatches(
                            accum_microbatches,
                            rng_namespace="prediction_step_monitor",
                        )
                        # Advance AdamW exactly once, including moments and
                        # weight decay, then extract its true parameter-space
                        # proposal before TIP changes the applied displacement.
                        self.optimizer.step()
                        parameters_after_raw_adamw = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_after_raw_adamw = (
                            self._prediction_loss_on_microbatches(
                                accum_microbatches,
                                rng_namespace="prediction_step_monitor",
                            )
                        )
                        raw_adamw_update = torch.cat([
                            (before - parameter).flatten()
                            for before, parameter in zip(
                                parameters_before, parameters, strict=True
                            )
                        ])
                        with torch.no_grad():
                            for parameter, before in zip(
                                parameters, parameters_before, strict=True
                            ):
                                parameter.copy_(before)

                        final_update = optimizer_update_modifier(
                            self, accum_microbatches, raw_adamw_update
                        )
                        offset = 0
                        with torch.no_grad():
                            if torch.equal(final_update, raw_adamw_update):
                                # Preserve AdamW's exact floating-point state
                                # when TIP is a no-op; reconstructing it as
                                # before-(before-after) accumulates roundoff.
                                for parameter, after in zip(
                                    parameters,
                                    parameters_after_raw_adamw,
                                    strict=True,
                                ):
                                    parameter.copy_(after)
                                offset = final_update.numel()
                            else:
                                for parameter in parameters:
                                    count = parameter.numel()
                                    parameter.add_(
                                        final_update[
                                            offset : offset + count
                                        ].view_as(parameter),
                                        alpha=-1.0,
                                    )
                                    offset += count
                        if offset != final_update.numel():
                            raise ValueError(
                                "TIP update size does not match imputer parameters"
                            )
                        pred_loss_after_tip_adamw = (
                            self._prediction_loss_on_microbatches(
                                accum_microbatches,
                                rng_namespace="prediction_step_monitor",
                            )
                        )
                        if (
                            optimizer_step_observer is not None
                            and pred_loss_before is not None
                            and pred_loss_after_raw_adamw is not None
                            and pred_loss_after_tip_adamw is not None
                        ):
                            optimizer_step_observer(
                                pred_loss_before,
                                pred_loss_after_raw_adamw,
                                pred_loss_after_tip_adamw,
                            )
                    else:
                        pred_loss_before = None
                        if optimizer_step_observer is not None:
                            pred_loss_before = (
                                self._prediction_loss_on_microbatches(
                                    accum_microbatches,
                                    rng_namespace="prediction_step_monitor",
                                )
                            )
                        self.optimizer.step()
                        if optimizer_step_observer is not None:
                            pred_loss_after = (
                                self._prediction_loss_on_microbatches(
                                    accum_microbatches,
                                    rng_namespace="prediction_step_monitor",
                                )
                            )
                            if (
                                pred_loss_before is not None
                                and pred_loss_after is not None
                            ):
                                optimizer_step_observer(
                                    pred_loss_before,
                                    pred_loss_after,
                                    pred_loss_after,
                                )
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

    def fit_tip(
        self, data_loader, phase_name, num_epoch, subspace_rank=4,
        epsilon=1e-3, confidence_delta=0.05, accum_steps=None,
        on_epoch_end=None, basis_mode="history", correction_mode="robust",
        gate_mode="ucb",
    ):
        """TIP with deterministic local-subspace gradient measurement."""
        if subspace_rank < 2:
            raise ValueError("subspace_rank must be at least 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must lie in (0, 1)")
        if basis_mode not in {"history", "random"}:
            raise ValueError("basis_mode must be 'history' or 'random'")
        if correction_mode not in {"robust", "point"}:
            raise ValueError("correction_mode must be 'robust' or 'point'")
        if gate_mode not in {"ucb", "mean"}:
            raise ValueError("gate_mode must be 'ucb' or 'mean'")
        if accum_steps is None:
            accum_steps = self.accum_steps
        self.logger.info(
            "TIP local subspace: target_rank=%d | "
            "finite_difference_queries_per_step=%d | "
            "step_monitor_queries_per_step=3 | total_queries_per_step=%d | "
            "epsilon=%g | confidence_delta=%g | basis_mode=%s | "
            "correction_mode=%s | gate_mode=%s | "
            "update_space=adamw_parameter_displacement | "
            "prediction_rng=common_random_numbers",
            subspace_rank, 2 * subspace_rank, 2 * subspace_rank + 3,
            epsilon, confidence_delta, basis_mode, correction_mode, gate_mode,
        )

        history: list[torch.Tensor] = []
        stats = {
            "steps": 0,
            "warmup": 0,
            "raw": 0,
            "projected": 0,
            "uncertain": 0,
            "rank": 0.0,
            "ucb": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "z_sem": 0.0,
            "adamw_update_norm": 0.0,
            "gp_subspace_norm": 0.0,
            "dot": 0.0,
            "cosine": 0.0,
            "candidate_correction_ratio": 0.0,
            "applied_correction_ratio": 0.0,
            "coordinate_variation": 0.0,
            "q_center": 0.0,
            "robust_margin": 0.0,
            "robust_solver_success": 0.0,
            "robust_fallback": 0.0,
        }
        prediction_change_stats = {
            "observed": 0,
            "before": 0.0,
            "raw_adamw_after": 0.0,
            "raw_adamw_delta": 0.0,
            "raw_adamw_relative_delta": 0.0,
            "raw_adamw_nonincrease": 0,
            "tip_adamw_after": 0.0,
            "tip_adamw_delta": 0.0,
            "tip_adamw_relative_delta": 0.0,
            "tip_adamw_nonincrease": 0,
            "projected_observed": 0,
            "projected_raw_adamw_delta": 0.0,
            "projected_raw_adamw_relative_delta": 0.0,
            "projected_raw_adamw_nonincrease": 0,
            "projected_tip_adamw_delta": 0.0,
            "projected_tip_adamw_relative_delta": 0.0,
            "projected_tip_adamw_nonincrease": 0,
        }
        last_step_projected = False

        def modifier(model, microbatches, raw_adamw_update):
            nonlocal last_step_projected
            # TIP now operates in the parameter-space direction that AdamW
            # actually proposed, after momentum, adaptive scaling, and decay.
            g_i = raw_adamw_update
            if basis_mode == "history":
                basis_candidates = [g_i, *reversed(history)]
            else:
                generator = torch.Generator(device=g_i.device)
                generator.manual_seed(
                    derive_seed(
                        model.base_seed,
                        f"tip_random_basis_{model.global_optim_step}",
                    )
                )
                basis_candidates = [
                    g_i,
                    *[
                        torch.randn(
                            g_i.shape,
                            generator=generator,
                            device=g_i.device,
                            dtype=g_i.dtype,
                        )
                        for _ in range(subspace_rank - 1)
                    ],
                ]
            basis = model._orthonormal_update_basis(
                basis_candidates, subspace_rank
            )
            effective_rank = basis.shape[1]
            update_coordinates = basis.T @ g_i

            coordinate_batches = []
            z_batches = []
            diagnostic_batches = []
            for rng_index, batch in enumerate(microbatches):
                coordinates, z_samples, diagnostics = (
                    model._compute_subspace_prediction_gradient(
                        batch["X_NA"], batch["X_NA_mask"], batch["X_N"],
                        batch["X_A_mask"], batch["Y_N"], batch["Y_N_mask"],
                        epsilon=epsilon,
                        basis=basis,
                        update_coordinates=update_coordinates,
                        rng_index=rng_index,
                    )
                )
                coordinate_batches.append(coordinates)
                z_batches.append(z_samples)
                if diagnostics:
                    diagnostic_batches.append(diagnostics)

            if diagnostic_batches:
                coordinate_samples = torch.cat([
                    item["coordinate_samples"] for item in diagnostic_batches
                ])
                prediction_coordinates = coordinate_samples.mean(dim=0)
                z_samples = coordinate_samples @ update_coordinates
            else:
                prediction_coordinates = torch.stack(
                    coordinate_batches
                ).mean(dim=0)
                z_samples = torch.cat(z_batches)
                coordinate_samples = prediction_coordinates.unsqueeze(0)
            dot = torch.dot(update_coordinates, prediction_coordinates)
            upper = model._student_t_upper_confidence_bound(
                z_samples, confidence_delta
            )
            gi_norm = torch.linalg.vector_norm(g_i)
            gp_norm = torch.linalg.vector_norm(prediction_coordinates)
            cosine = dot / (gi_norm * gp_norm).clamp_min(1e-12)
            raw_conflict = dot.item() < 0.0
            rank_ready = effective_rank >= subspace_rank
            gate_conflict = (
                upper.item() < 0.0 if gate_mode == "ucb" else dot.item() < 0.0
            )
            project = rank_ready and gate_conflict and gp_norm.item() > 1e-12
            last_step_projected = project

            candidate_coordinates = update_coordinates
            robust_diagnostics = {
                "success": 0.0,
                "margin": 0.0,
                "fallback": 0.0,
            }
            if project:
                if correction_mode == "robust":
                    candidate_coordinates, robust_diagnostics = (
                        model._robust_subspace_projection(
                            update_coordinates,
                            coordinate_samples,
                            confidence_delta,
                        )
                    )
                else:
                    denominator = torch.dot(
                        prediction_coordinates, prediction_coordinates
                    ).clamp_min(1e-12)
                    candidate_coordinates = (
                        update_coordinates
                        - dot / denominator * prediction_coordinates
                    )
                    robust_diagnostics = {
                        "success": 1.0,
                        "margin": 0.0,
                        "fallback": 0.0,
                    }
            candidate = g_i + basis @ (
                candidate_coordinates - update_coordinates
            )
            correction = candidate - g_i
            correction_ratio = (
                torch.linalg.vector_norm(correction) / gi_norm.clamp_min(1e-12)
            )

            stats["steps"] += 1
            stats["warmup"] += int(not rank_ready)
            stats["raw"] += int(raw_conflict)
            stats["projected"] += int(project)
            stats["uncertain"] += int(rank_ready and raw_conflict and not project)
            stats["rank"] += effective_rank
            stats["ucb"] += upper.item()
            stats["z_mean"] += z_samples.mean().item()
            z_std = z_samples.std(unbiased=z_samples.numel() > 1)
            stats["z_std"] += z_std.item()
            stats["z_sem"] += (z_std / math.sqrt(z_samples.numel())).item()
            stats["adamw_update_norm"] += gi_norm.item()
            stats["gp_subspace_norm"] += gp_norm.item()
            stats["dot"] += dot.item()
            stats["cosine"] += cosine.item()
            stats["candidate_correction_ratio"] += correction_ratio.item()
            stats["applied_correction_ratio"] += (
                correction_ratio.item() if project else 0.0
            )
            if diagnostic_batches:
                coordinate_mean = coordinate_samples.mean(dim=0)
                coordinate_spread = (
                    coordinate_samples.std(dim=0, unbiased=True)
                    if coordinate_samples.shape[0] > 1
                    else torch.zeros_like(coordinate_mean)
                )
                stats["coordinate_variation"] += (
                    coordinate_spread.norm()
                    / coordinate_mean.norm().clamp_min(1e-12)
                ).item()
                stats["q_center"] += torch.cat([
                    item["q_center"].flatten() for item in diagnostic_batches
                ]).mean().item()
            stats["robust_margin"] += robust_diagnostics["margin"]
            stats["robust_solver_success"] += robust_diagnostics["success"]
            stats["robust_fallback"] += robust_diagnostics["fallback"]

            # The local model should follow directions actually presented to
            # the optimizer, including robust corrections.
            history.append(candidate.detach().clone() if project else g_i.detach().clone())
            history_limit = subspace_rank - 1
            if len(history) > history_limit:
                del history[:-history_limit]
            return candidate if project else g_i

        def observe_optimizer_step(
            pred_loss_before: float,
            pred_loss_after_raw_adamw: float,
            pred_loss_after_tip_adamw: float,
        ) -> None:
            self.record_prediction_optimizer_step(
                phase_name=phase_name,
                pred_loss_before=pred_loss_before,
                pred_loss_after_raw_adamw=pred_loss_after_raw_adamw,
                pred_loss_after_applied_update=pred_loss_after_tip_adamw,
                projection_applied=last_step_projected,
            )
            denominator = max(abs(pred_loss_before), 1e-12)
            raw_adamw_delta = pred_loss_after_raw_adamw - pred_loss_before
            tip_adamw_delta = pred_loss_after_tip_adamw - pred_loss_before
            raw_adamw_relative_delta = raw_adamw_delta / denominator
            tip_adamw_relative_delta = tip_adamw_delta / denominator
            prediction_change_stats["observed"] += 1
            prediction_change_stats["before"] += pred_loss_before
            prediction_change_stats["raw_adamw_after"] += (
                pred_loss_after_raw_adamw
            )
            prediction_change_stats["raw_adamw_delta"] += raw_adamw_delta
            prediction_change_stats["raw_adamw_relative_delta"] += (
                raw_adamw_relative_delta
            )
            prediction_change_stats["raw_adamw_nonincrease"] += int(
                raw_adamw_delta <= 0.0
            )
            prediction_change_stats["tip_adamw_after"] += (
                pred_loss_after_tip_adamw
            )
            prediction_change_stats["tip_adamw_delta"] += tip_adamw_delta
            prediction_change_stats["tip_adamw_relative_delta"] += (
                tip_adamw_relative_delta
            )
            prediction_change_stats["tip_adamw_nonincrease"] += int(
                tip_adamw_delta <= 0.0
            )
            if last_step_projected:
                prediction_change_stats["projected_observed"] += 1
                prediction_change_stats[
                    "projected_raw_adamw_delta"
                ] += raw_adamw_delta
                prediction_change_stats[
                    "projected_raw_adamw_relative_delta"
                ] += raw_adamw_relative_delta
                prediction_change_stats[
                    "projected_raw_adamw_nonincrease"
                ] += int(raw_adamw_delta <= 0.0)
                prediction_change_stats[
                    "projected_tip_adamw_delta"
                ] += tip_adamw_delta
                prediction_change_stats[
                    "projected_tip_adamw_relative_delta"
                ] += tip_adamw_relative_delta
                prediction_change_stats[
                    "projected_tip_adamw_nonincrease"
                ] += int(tip_adamw_delta <= 0.0)

        def epoch_end(epoch, train_loss, model):
            steps = stats["steps"]
            if steps:
                projected = max(stats["projected"], 1)
                averages = {
                    key: value / steps
                    for key, value in stats.items()
                    if key not in {
                        "steps", "warmup", "raw", "projected", "uncertain"
                    }
                }
                model.logger.info(
                    "TIP subspace: mean_rank=%.2f/%d | warmup_rate=%.4f | "
                    "raw_conflict_rate=%.4f | projection_rate=%.4f | "
                    "uncertain_conflict_rate=%.4f | steps=%d",
                    averages["rank"], subspace_rank, stats["warmup"] / steps,
                    stats["raw"] / steps, stats["projected"] / steps,
                    stats["uncertain"] / steps, steps,
                )
                model.logger.info(
                    "TIP confidence: mean_ucb=%.6g | z_mean=%.6g | "
                    "z_std=%.6g | z_sem=%.6g | coordinate_variation=%.6g | "
                    "q_center=%.6g",
                    averages["ucb"], averages["z_mean"], averages["z_std"],
                    averages["z_sem"], averages["coordinate_variation"],
                    averages["q_center"],
                )
                model.logger.info(
                    "TIP geometry: adamw_update_norm=%.6g | "
                    "gp_subspace_norm=%.6g | "
                    "dot=%.6g | cosine=%.6g | "
                    "candidate_correction_ratio=%.6g | "
                    "applied_correction_ratio=%.6g | robust_margin=%.6g | "
                    "robust_solver_success=%.4f | robust_fallback=%.4f",
                    averages["adamw_update_norm"],
                    averages["gp_subspace_norm"],
                    averages["dot"], averages["cosine"],
                    averages["candidate_correction_ratio"],
                    averages["applied_correction_ratio"],
                    stats["robust_margin"] / projected,
                    stats["robust_solver_success"] / projected,
                    stats["robust_fallback"] / projected,
                )
                observed = max(prediction_change_stats["observed"], 1)
                projected_observed = max(
                    prediction_change_stats["projected_observed"], 1
                )
                prediction_change_values = {
                    "pred_loss_before": (
                        prediction_change_stats["before"] / observed
                    ),
                    "raw_adamw_pred_loss_after": (
                        prediction_change_stats["raw_adamw_after"] / observed
                    ),
                    "raw_adamw_pred_loss_delta": (
                        prediction_change_stats["raw_adamw_delta"] / observed
                    ),
                    "raw_adamw_pred_loss_relative_delta": (
                        prediction_change_stats["raw_adamw_relative_delta"]
                        / observed
                    ),
                    "raw_adamw_pred_loss_nonincrease_rate": (
                        prediction_change_stats["raw_adamw_nonincrease"]
                        / observed
                    ),
                    "tip_adamw_pred_loss_after": (
                        prediction_change_stats["tip_adamw_after"] / observed
                    ),
                    "tip_adamw_pred_loss_delta": (
                        prediction_change_stats["tip_adamw_delta"] / observed
                    ),
                    "tip_adamw_pred_loss_relative_delta": (
                        prediction_change_stats["tip_adamw_relative_delta"]
                        / observed
                    ),
                    "tip_adamw_pred_loss_nonincrease_rate": (
                        prediction_change_stats["tip_adamw_nonincrease"]
                        / observed
                    ),
                    "projected_raw_adamw_pred_loss_delta": (
                        prediction_change_stats[
                            "projected_raw_adamw_delta"
                        ] / projected_observed
                    ),
                    "projected_raw_adamw_pred_loss_relative_delta": (
                        prediction_change_stats[
                            "projected_raw_adamw_relative_delta"
                        ] / projected_observed
                    ),
                    "projected_raw_adamw_pred_loss_nonincrease_rate": (
                        prediction_change_stats[
                            "projected_raw_adamw_nonincrease"
                        ] / projected_observed
                    ),
                    "projected_tip_adamw_pred_loss_delta": (
                        prediction_change_stats["projected_tip_adamw_delta"]
                        / projected_observed
                    ),
                    "projected_tip_adamw_pred_loss_relative_delta": (
                        prediction_change_stats[
                            "projected_tip_adamw_relative_delta"
                        ]
                        / projected_observed
                    ),
                    "projected_tip_adamw_pred_loss_nonincrease_rate": (
                        prediction_change_stats[
                            "projected_tip_adamw_nonincrease"
                        ]
                        / projected_observed
                    ),
                }
                prediction_change_values["tip_projection_gain"] = (
                    prediction_change_values["raw_adamw_pred_loss_delta"]
                    - prediction_change_values["tip_adamw_pred_loss_delta"]
                )
                prediction_change_values["projected_tip_projection_gain"] = (
                    prediction_change_values[
                        "projected_raw_adamw_pred_loss_delta"
                    ]
                    - prediction_change_values[
                        "projected_tip_adamw_pred_loss_delta"
                    ]
                )
                model.logger.info(
                    "TIP raw AdamW proposal: pred_loss_before=%.6g | "
                    "pred_loss_after=%.6g | pred_loss_delta=%.6g | "
                    "pred_loss_relative_delta=%.6g | "
                    "pred_loss_nonincrease_rate=%.4f | "
                    "projected_pred_loss_delta=%.6g | "
                    "projected_pred_loss_relative_delta=%.6g | "
                    "projected_pred_loss_nonincrease_rate=%.4f",
                    prediction_change_values["pred_loss_before"],
                    prediction_change_values["raw_adamw_pred_loss_after"],
                    prediction_change_values["raw_adamw_pred_loss_delta"],
                    prediction_change_values[
                        "raw_adamw_pred_loss_relative_delta"
                    ],
                    prediction_change_values[
                        "raw_adamw_pred_loss_nonincrease_rate"
                    ],
                    prediction_change_values[
                        "projected_raw_adamw_pred_loss_delta"
                    ],
                    prediction_change_values[
                        "projected_raw_adamw_pred_loss_relative_delta"
                    ],
                    prediction_change_values[
                        "projected_raw_adamw_pred_loss_nonincrease_rate"
                    ],
                )
                model.logger.info(
                    "TIP projected AdamW step: pred_loss_after=%.6g | "
                    "pred_loss_delta=%.6g | pred_loss_relative_delta=%.6g | "
                    "pred_loss_nonincrease_rate=%.4f | "
                    "projected_pred_loss_delta=%.6g | "
                    "projected_pred_loss_relative_delta=%.6g | "
                    "projected_pred_loss_nonincrease_rate=%.4f | "
                    "tip_projection_gain=%.6g | "
                    "projected_tip_projection_gain=%.6g | "
                    "observed_steps=%d | projected_observed_steps=%d",
                    prediction_change_values["tip_adamw_pred_loss_after"],
                    prediction_change_values["tip_adamw_pred_loss_delta"],
                    prediction_change_values[
                        "tip_adamw_pred_loss_relative_delta"
                    ],
                    prediction_change_values[
                        "tip_adamw_pred_loss_nonincrease_rate"
                    ],
                    prediction_change_values[
                        "projected_tip_adamw_pred_loss_delta"
                    ],
                    prediction_change_values[
                        "projected_tip_adamw_pred_loss_relative_delta"
                    ],
                    prediction_change_values[
                        "projected_tip_adamw_pred_loss_nonincrease_rate"
                    ],
                    prediction_change_values["tip_projection_gain"],
                    prediction_change_values[
                        "projected_tip_projection_gain"
                    ],
                    prediction_change_stats["observed"],
                    prediction_change_stats["projected_observed"],
                )
                tb_values = {
                    "effective_rank": averages["rank"],
                    "warmup_rate": stats["warmup"] / steps,
                    "raw_conflict_rate": stats["raw"] / steps,
                    "projection_rate": stats["projected"] / steps,
                    "uncertain_conflict_rate": stats["uncertain"] / steps,
                    **averages,
                    "robust_margin": stats["robust_margin"] / projected,
                    "robust_solver_success": (
                        stats["robust_solver_success"] / projected
                    ),
                    "robust_fallback": stats["robust_fallback"] / projected,
                    **prediction_change_values,
                    "prediction_change_observed_steps": (
                        prediction_change_stats["observed"]
                    ),
                    "prediction_change_projected_observed_steps": (
                        prediction_change_stats["projected_observed"]
                    ),
                }
                for name, value in tb_values.items():
                    model.tb_summary_writer.add_scalar(
                        f"tip/{phase_name}/{name}", value, epoch
                    )
            for key in stats:
                stats[key] = 0 if key in {
                    "steps", "warmup", "raw", "projected", "uncertain"
                } else 0.0
            for key in prediction_change_stats:
                prediction_change_stats[key] = 0 if key in {
                    "observed", "raw_adamw_nonincrease",
                    "tip_adamw_nonincrease", "projected_observed",
                    "projected_raw_adamw_nonincrease",
                    "projected_tip_adamw_nonincrease",
                } else 0.0
            return on_epoch_end(epoch, train_loss, model) if on_epoch_end else False

        return self.fit(
            data_loader, phase_name, num_epoch, accum_steps=accum_steps,
            on_epoch_end=epoch_end, optimizer_update_modifier=modifier,
            optimizer_step_observer=observe_optimizer_step,
        )
