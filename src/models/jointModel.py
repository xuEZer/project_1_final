"""Joint training wrapper for imputation and prediction models.

The imputer is trained via its reconstruction / imputation loss.  The
predictor is **inference-only** — it runs inside ``torch.no_grad()``,
does not produce a loss, and never participates in backpropagation.
"""

from __future__ import annotations

import collections.abc
import logging
from collections.abc import Callable
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.rng import derive_seed, torch_seed_scope


optimizer_dict = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}


def _discard_metric(name: str, value: float, step: int) -> None:
    """Default sink for callers that do not collect scalar diagnostics."""


class JointModel(nn.Module):
    """Wrap an imputer (trainable) and predictor (inference-only)."""

    def __init__(
        self,
        imputer: nn.Module,
        predictor: nn.Module,
        optimizer_name: str,
        lr: float,
        device: str,
        record_metric: Callable[[str, float, int], None] | None,
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
            record_metric: Sink for scalar diagnostics, called as
                ``(name, value, step)``.  ``None`` drops them, which is what the
                auto batch-size probe wants.
            logger: Logger instance.
        """
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor
        self.record_metric = record_metric or _discard_metric
        self.logger = logger
        self.device = device
        self.optimizer_name = optimizer_name

        self.global_epoch = 0
        self.global_optim_step = 0
        self.accum_steps = 1
        self.micro_batch_size = 1
        self.non_blocking_transfer = str(device).startswith("cuda")
        self.base_seed = 0

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
        self.predictor = predictor
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()

    def configure_base_seed(self, base_seed: int) -> None:
        """Record the experiment seed for the deterministic per-step bases.

        Training randomness comes from the process-global generator that expman
        seeds and restores around every Stage, so nothing is seeded here.  The
        ablations that need the *same* random basis at a given step build it
        from an explicit ``torch.Generator`` keyed on this seed instead of
        consuming the shared stream.
        """
        self.base_seed = int(base_seed)

    def _synchronize_for_timing(self) -> None:
        """Wait for queued accelerator work before reading a wall-clock timer."""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

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

        pred_data = None
        if run_predictor:
            with torch.no_grad():
                predictor_output = self.predictor(imputed_X)
            pred_data = predictor_output.get("pred_data", None)

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
    def compute_val_prediction_loss(self, val_loader: DataLoader) -> float:
        """Average prediction loss on the validation set.

        Same weighting convention as :meth:`compute_val_loss` (batch sizes, and
        here the number of observed targets), so the last partial batch cannot
        distort the metric.  It reports how well the frozen forecaster does on
        the current imputer state.
        """
        if val_loader is None:
            return float("inf")
        was_training = self.imputer.training
        self.imputer.eval()
        weighted = 0.0
        count = 0.0
        try:
            with torch.no_grad():
                for batch in val_loader:
                    (
                        X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask,
                    ) = self._move_batch_to_device(batch)
                    output = self.forward(
                        X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                        run_predictor=True,
                    )
                    if output["pred_loss"] is None:
                        continue
                    n = (
                        float((1.0 - Y_N_mask).sum())
                        if Y_N_mask is not None else float(Y_N.numel())
                    )
                    weighted += output["pred_loss"].item() * n
                    count += n
        finally:
            self.imputer.train(was_training)
        return weighted / max(count, 1.0)

    def compute_val_loss(self, val_loader: DataLoader) -> float:
        """Average imputation loss on the validation set.

        Losses are weighted by batch size so the last (potentially
        smaller) batch does not distort the validation metric.
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

    def _move_batch_to_device(
        self,
        batch: collections.abc.Iterable[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            t.to(self.device, non_blocking=self.non_blocking_transfer) for t in batch
        )

    @staticmethod
    def _capture_training_modes(module: nn.Module) -> dict[nn.Module, bool]:
        return {submodule: submodule.training for submodule in module.modules()}

    @staticmethod
    def _restore_training_modes(training_modes: dict[nn.Module, bool]) -> None:
        for module, is_training in training_modes.items():
            module.train(is_training)

    @staticmethod
    def _orthonormal_update_basis(
        vectors: list[torch.Tensor],
        max_rank: int,
        relative_tolerance: float = 1e-6,
    ) -> torch.Tensor | None:
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
            return None
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
        comparison measures only the parameter update, not dropout noise.
        Each micro-batch is forwarded separately, bounding activation memory
        to one micro-batch instead of the whole accumulation window.  Every
        micro-batch enters the same per-step seed scope, so before/after share
        identical sampling noise and their difference stays unbiased (CRN).
        The per-micro-batch MSEs are aggregated by observed-target count,
        which reproduces the concatenated-batch masked MSE.
        """
        if not microbatches:
            return None

        training_modes = self._capture_training_modes(self.imputer)
        weighted_error_sum = 0.0
        target_count = 0.0
        try:
            self.imputer.eval()
            for microbatch in microbatches:
                inputs = (
                    microbatch["X_NA"],
                    microbatch["X_NA_mask"],
                    microbatch["X_N"],
                    microbatch["X_A_mask"],
                    microbatch["Y_N"],
                    microbatch["Y_N_mask"],
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
                if pred_loss is None:
                    continue
                y_mask = microbatch["Y_N_mask"]
                count = (
                    float((1.0 - y_mask).sum())
                    if y_mask is not None
                    else float(microbatch["Y_N"].numel())
                )
                weighted_error_sum += pred_loss.item() * count
                target_count += count
            if target_count <= 0:
                return None
            return weighted_error_sum / target_count
        finally:
            self._restore_training_modes(training_modes)

    @torch.no_grad()
    def _prediction_loss_on_loader(
        self,
        loader: DataLoader,
        rng_namespace: str | None = None,
    ) -> float | None:
        """Evaluate prediction MSE over an entire loader, streaming by batch."""
        training_modes = self._capture_training_modes(self.imputer)
        weighted_error_sum = 0.0
        target_count = 0.0
        try:
            self.imputer.eval()
            for batch_index, batch in enumerate(loader):
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                    self._move_batch_to_device(batch)
                )
                inputs = (X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask)
                if rng_namespace is None:
                    output = self.forward(*inputs, run_predictor=True)
                else:
                    with torch_seed_scope(
                        self.base_seed,
                        rng_namespace,
                        self.global_optim_step,
                        batch_index,
                    ):
                        output = self.forward(*inputs, run_predictor=True)
                pred_loss = output["pred_loss"]
                if pred_loss is None:
                    continue
                count = float((1.0 - Y_N_mask).sum())
                weighted_error_sum += pred_loss.item() * count
                target_count += count
            if target_count <= 0:
                return None
            return weighted_error_sum / target_count
        finally:
            self._restore_training_modes(training_modes)

    def record_prediction_optimizer_step(
        self,
        *,
        pred_loss_before: float,
        pred_loss_after_raw_adamw: float,
        pred_loss_after_applied_update: float,
        projection_applied: bool = False,
    ) -> None:
        step = self.global_optim_step + 1
        denominator = max(abs(pred_loss_before), 1e-12)
        raw_delta = pred_loss_after_raw_adamw - pred_loss_before
        applied_delta = pred_loss_after_applied_update - pred_loss_before
        projection_gain = (
            pred_loss_after_raw_adamw - pred_loss_after_applied_update
        )
        values = {
            "pred_loss_before": pred_loss_before,
            "pred_loss_delta": applied_delta,
            "pred_loss_decrease": -applied_delta,
            "pred_loss_relative_delta": applied_delta / denominator,
            "raw_adamw_pred_loss_delta": raw_delta,
            "raw_adamw_pred_loss_decrease": -raw_delta,
            "raw_adamw_pred_loss_relative_delta": raw_delta / denominator,
            "tip_projection_gain": projection_gain,
            # The gate's decision, not ``tip_projection_gain != 0``: a
            # projection that changes nothing numerically is still a step the
            # gate selected, and the per-projection statistics need it.
            "projection_applied": float(projection_applied),
        }
        for name, value in values.items():
            self.record_metric(f"prediction_step/{name}", value, step)
        self.logger.info(
            "Prediction optimizer step: step=%d | "
            "pred_loss_before=%.9g | raw_adamw_pred_loss_delta=%.9g | "
            "pred_loss_delta=%.9g | pred_loss_relative_delta=%.9g | "
            "tip_projection_gain=%.9g | projection_applied=%d | "
            "pred_loss_decrease=%.9g | raw_adamw_pred_loss_decrease=%.9g",
            step,
            pred_loss_before,
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

    @staticmethod
    def _robust_cone_margin(
        coordinates: torch.Tensor,
        coordinate_samples: torch.Tensor,
        delta: float,
    ) -> float:
        """Value of the robust cone constraint at ``coordinates``.

        Mirrors the constraint used by :meth:`_robust_subspace_projection`, so
        the task anchor can be backtracked until it stays inside the cone
        instead of silently leaving the feasible set.
        """
        if coordinate_samples.ndim != 2 or coordinate_samples.shape[0] < 2:
            return 0.0
        trial = coordinates.detach().double().cpu().numpy()
        samples = coordinate_samples.detach().double().cpu().numpy()
        rank = samples.shape[1]
        mean = np.mean(samples, axis=0)
        covariance = np.cov(samples, rowvar=False, ddof=1) / samples.shape[0]
        covariance = np.atleast_2d(covariance)
        ridge = max(float(np.trace(covariance)) / max(rank, 1) * 1e-8, 1e-12)
        covariance = covariance + ridge * np.eye(rank)
        beta = float(student_t.ppf(1.0 - delta, df=samples.shape[0] - 1))
        variance = max(float(trial @ covariance @ trial), 0.0)
        return float(mean @ trial - beta * math.sqrt(variance))

    def _sync_first_moment_to_applied_update(
        self,
        parameters: list[torch.nn.Parameter],
        parameters_before: list[torch.Tensor],
        applied_update: torch.Tensor,
    ) -> None:
        """Make Adam's first moment consistent with the displacement actually applied.

        TIP replaces the optimizer's proposed displacement *after*
        ``optimizer.step()`` has already advanced the moment estimates using the
        uncorrected imputation gradient.  The state therefore keeps implying the
        raw step, so the next proposal re-injects the component TIP just removed,
        and the paired B1/B2 divergence random-walks instead of drifting.

        AdamW proposes ``d = lr * m_hat / (sqrt(v_hat) + eps) + lr * wd * theta``,
        so the state that implies exactly ``applied_update`` is

            m = (1 - beta1**t) * (sqrt(v_hat) + eps) / lr
                * (applied_update - lr * wd * theta_before).

        Only ``exp_avg`` is rewritten.  ``exp_avg_sq`` is Adam's curvature
        estimate and must remain a statistic of the real gradients, otherwise the
        per-coordinate step size stops being calibrated.

        The applied displacement is unchanged by this call, so the per-step
        theorems (minimum-change projection, subspace preservation, covariance-set
        robustness) still describe the applied update verbatim; only the proposal
        at the next step changes, and no theorem constrains that proposal.
        """
        group = self.optimizer.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group.get("weight_decay", 0.0)
        offset = 0
        with torch.no_grad():
            for parameter, before in zip(
                parameters, parameters_before, strict=True
            ):
                count = parameter.numel()
                applied = applied_update[offset : offset + count].view_as(parameter)
                offset += count
                state = self.optimizer.state.get(parameter)
                if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
                    continue
                step = state["step"]
                step = int(step.item()) if torch.is_tensor(step) else int(step)
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                v_hat = state["exp_avg_sq"] / bias_correction2
                target = applied
                if weight_decay:
                    target = applied - lr * weight_decay * before
                state["exp_avg"].copy_(
                    (bias_correction1 / lr) * (v_hat.sqrt() + eps) * target
                )

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
        counterfactual_update_generator: collections.abc.Callable | None = None,
        counterfactual_step_observer: collections.abc.Callable | None = None,
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
                accumulated micro-batches, and the actual optimizer update vector
                ``theta_before - theta_after``. It returns the update vector
                that will be applied to the parameters.
            optimizer_step_observer: Optional callback receiving the complete
                training-set prediction loss before the update, after the raw
                optimizer proposal, and after the TIP-projected update.
                Enabling it adds three full training-loader evaluations per
                optimizer step.
            counterfactual_update_generator: Optional callback that returns a
                mapping of named candidate updates. Every candidate is
                evaluated from the same pre-step parameters, while only the
                raw optimizer proposal is finally applied.

        Returns:
            Final training loss from the last epoch in this phase.
        """
        if accum_steps is None:
            accum_steps = getattr(self, "accum_steps", 1)

        train_loader = data_loader["train"]
        prediction_evaluation_loader = data_loader.get("train_eval", train_loader)
        final_train_loss = float("inf")
        self.phase_epoch = 0
        measure_wall_clock = phase_name == "rq4"

        while self.phase_epoch < num_epoch:
            epoch_number = self.global_epoch + 1
            if measure_wall_clock:
                self._synchronize_for_timing()
                epoch_started = time.perf_counter()
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
                    if measure_wall_clock:
                        self._synchronize_for_timing()
                        optimizer_step_started = time.perf_counter()
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
                    if measure_wall_clock:
                        self._synchronize_for_timing()
                        training_prefix_seconds = (
                            time.perf_counter() - optimizer_step_started
                        )

                    if counterfactual_update_generator is not None:
                        parameters = list(self.imputer.parameters())
                        parameters_before = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_before = self._prediction_loss_on_loader(
                            prediction_evaluation_loader,
                            rng_namespace="prediction_step_monitor",
                        )
                        if measure_wall_clock:
                            self._synchronize_for_timing()
                            optimizer_started = time.perf_counter()
                        self.optimizer.step()
                        if measure_wall_clock:
                            self._synchronize_for_timing()
                            optimizer_seconds = (
                                time.perf_counter() - optimizer_started
                            )
                        parameters_after_raw_adamw = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_after_raw_adamw = (
                            self._prediction_loss_on_loader(
                                prediction_evaluation_loader,
                                rng_namespace="prediction_step_monitor",
                            )
                        )
                        if measure_wall_clock:
                            base_step_seconds = (
                                training_prefix_seconds + optimizer_seconds
                            )
                            self.logger.info(
                                "Counterfactual base step timing: phase=%s | "
                                "epoch=%d | step=%d | wall_seconds=%.9f",
                                phase_name,
                                epoch_number,
                                self.global_optim_step + 1,
                                base_step_seconds,
                            )
                            self.record_metric(
                                "timing/base_step_seconds",
                                base_step_seconds,
                                self.global_optim_step + 1,
                            )
                        raw_adamw_update = torch.cat([
                            (before - after).flatten()
                            for before, after in zip(
                                parameters_before,
                                parameters_after_raw_adamw,
                                strict=True,
                            )
                        ])
                        with torch.no_grad():
                            for parameter, before in zip(
                                parameters, parameters_before, strict=True
                            ):
                                parameter.copy_(before)
                        candidates = counterfactual_update_generator(
                            self, accum_microbatches, raw_adamw_update
                        )
                        candidate_losses: dict[str, float | None] = {}
                        evaluated_candidates: dict[int, float | None] = {}
                        for name, candidate in candidates.items():
                            if torch.equal(candidate, raw_adamw_update):
                                candidate_losses[name] = pred_loss_after_raw_adamw
                                continue
                            candidate_id = id(candidate)
                            if candidate_id in evaluated_candidates:
                                candidate_losses[name] = evaluated_candidates[candidate_id]
                                continue
                            offset = 0
                            with torch.no_grad():
                                for parameter, before in zip(
                                    parameters, parameters_before, strict=True
                                ):
                                    parameter.copy_(before)
                                for parameter in parameters:
                                    count = parameter.numel()
                                    parameter.add_(
                                        candidate[offset : offset + count].view_as(
                                            parameter
                                        ),
                                        alpha=-1.0,
                                    )
                                    offset += count
                            if offset != candidate.numel():
                                raise ValueError(
                                    f"Counterfactual update {name} has the wrong size"
                                )
                            candidate_losses[name] = (
                                self._prediction_loss_on_loader(
                                    prediction_evaluation_loader,
                                    rng_namespace="prediction_step_monitor",
                                )
                            )
                            evaluated_candidates[candidate_id] = candidate_losses[name]
                        # The only realised update is the exact raw AdamW
                        # proposal, including its original floating-point state.
                        with torch.no_grad():
                            for parameter, after in zip(
                                parameters,
                                parameters_after_raw_adamw,
                                strict=True,
                            ):
                                parameter.copy_(after)
                        if (
                            counterfactual_step_observer is not None
                            and pred_loss_before is not None
                            and pred_loss_after_raw_adamw is not None
                        ):
                            counterfactual_step_observer(
                                pred_loss_before,
                                pred_loss_after_raw_adamw,
                                candidate_losses,
                            )
                    elif optimizer_update_modifier is not None:
                        parameters = list(self.imputer.parameters())
                        parameters_before = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_before = self._prediction_loss_on_loader(
                            prediction_evaluation_loader,
                            rng_namespace="prediction_step_monitor",
                        )
                        # Advance the optimizer exactly once, including any
                        # state and weight decay, then extract its true parameter-space
                        # proposal before TIP changes the applied displacement.
                        self.optimizer.step()
                        parameters_after_raw_adamw = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        pred_loss_after_raw_adamw = (
                            self._prediction_loss_on_loader(
                                prediction_evaluation_loader,
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
                                # Preserve the optimizer's exact floating-point state
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
                        # Keep the optimizer's first moment consistent with the
                        # displacement actually applied.  Without this, AdamW's
                        # state still implies the raw proposal, so the next step
                        # re-injects the component TIP just removed and the
                        # correction never accumulates.
                        if not torch.equal(final_update, raw_adamw_update):
                            self._sync_first_moment_to_applied_update(
                                parameters, parameters_before, final_update
                            )
                        pred_loss_after_tip_adamw = (
                            self._prediction_loss_on_loader(
                                prediction_evaluation_loader,
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
                                self._prediction_loss_on_loader(
                                    prediction_evaluation_loader,
                                    rng_namespace="prediction_step_monitor",
                                )
                            )
                        self.optimizer.step()
                        if optimizer_step_observer is not None:
                            pred_loss_after = (
                                self._prediction_loss_on_loader(
                                    prediction_evaluation_loader,
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

            if measure_wall_clock:
                # Before the epoch observer: the observer scores both splits
                # with the predictor, and that work is not part of the training
                # pass this metric is meant to time.
                self._synchronize_for_timing()
                epoch_seconds = time.perf_counter() - epoch_started
                self.logger.info(
                    "Epoch wall clock: phase=%s | epoch=%d | wall_seconds=%.9f",
                    phase_name,
                    epoch_number,
                    epoch_seconds,
                )
                self.record_metric(
                    "timing/epoch_seconds", epoch_seconds, epoch_number
                )

            if on_epoch_end is not None:
                on_epoch_end(
                    self.global_epoch + 1,
                    avg_train_loss,
                    self,
                )

            self.phase_epoch += 1
            self.global_epoch += 1

        self.logger.info(
            f"Finished Phase: {phase_name}. Final Train Loss: {final_train_loss:.6f}"
        )

        return final_train_loss

    def fit_rq3_counterfactual(
        self,
        data_loader,
        phase_name,
        num_epoch,
        subspace_rank=4,
        epsilon=1e-3,
        confidence_delta=0.05,
        accum_steps=None,
        on_epoch_end=None,
    ):
        """Evaluate B2--B5 at every step while realising only the B1 update."""
        if subspace_rank < 2:
            raise ValueError("subspace_rank must be at least 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must lie in (0, 1)")
        if accum_steps is None:
            accum_steps = self.accum_steps
        prediction_gradient_loader = data_loader.get(
            "train_eval", data_loader["train"]
        )

        history: list[torch.Tensor] = []
        step_metadata: dict[str, dict[str, bool]] = {}

        def measure(model, microbatches, basis, update_coordinates):
            coordinate_batches = []
            diagnostic_batches = []
            for rng_index, raw_batch in enumerate(prediction_gradient_loader):
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                    model._move_batch_to_device(raw_batch)
                )
                coordinates, _, diagnostics = (
                    model._compute_subspace_prediction_gradient(
                        X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                        epsilon=epsilon,
                        basis=basis,
                        update_coordinates=update_coordinates,
                        rng_index=rng_index,
                    )
                )
                coordinate_batches.append(coordinates)
                if diagnostics:
                    diagnostic_batches.append(diagnostics)
            if diagnostic_batches:
                coordinate_samples = torch.cat([
                    item["coordinate_samples"] for item in diagnostic_batches
                ])
                prediction_coordinates = coordinate_samples.mean(dim=0)
            else:
                prediction_coordinates = torch.stack(coordinate_batches).mean(dim=0)
                coordinate_samples = prediction_coordinates.unsqueeze(0)
            z_samples = coordinate_samples @ update_coordinates
            return prediction_coordinates, coordinate_samples, z_samples

        def candidate(
            g_i,
            basis,
            prediction_coordinates,
            coordinate_samples,
            z_samples,
            *,
            correction_mode,
            gate_mode,
        ):
            update_coordinates = basis.T @ g_i
            dot = torch.dot(update_coordinates, prediction_coordinates)
            upper = self._student_t_upper_confidence_bound(
                z_samples, confidence_delta
            )
            conflict = upper.item() < 0.0 if gate_mode == "ucb" else dot.item() < 0.0
            project = (
                basis.shape[1] >= subspace_rank
                and conflict
                and torch.linalg.vector_norm(prediction_coordinates).item() > 1e-12
            )
            candidate_coordinates = update_coordinates
            if project and correction_mode == "robust":
                candidate_coordinates, _ = self._robust_subspace_projection(
                    update_coordinates, coordinate_samples, confidence_delta
                )
            elif project:
                denominator = torch.dot(
                    prediction_coordinates, prediction_coordinates
                ).clamp_min(1e-12)
                candidate_coordinates = (
                    update_coordinates
                    - dot / denominator * prediction_coordinates
                )
            update = g_i + basis @ (candidate_coordinates - update_coordinates)
            return update if project else g_i, {
                "conflict": conflict,
                "projected": project,
            }

        def generate(model, microbatches, raw_adamw_update):
            nonlocal step_metadata
            g_i = raw_adamw_update
            history_basis = model._orthonormal_update_basis(
                [g_i, *reversed(history)], subspace_rank
            )
            generator = torch.Generator(device=g_i.device)
            generator.manual_seed(
                derive_seed(
                    model.base_seed,
                    f"rq3_random_basis_{model.global_optim_step}",
                )
            )
            random_basis = model._orthonormal_update_basis(
                [
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
                ],
                subspace_rank,
            )

            outputs: dict[str, torch.Tensor] = {}
            step_metadata = {}
            if history_basis is None:
                for stage in ("B2", "B4", "B5"):
                    outputs[stage] = g_i
                    step_metadata[stage] = {"conflict": False, "projected": False}
            else:
                history_coordinates = history_basis.T @ g_i
                pred, samples, z_samples = measure(
                    model, microbatches, history_basis, history_coordinates
                )
                for stage, correction_mode, gate_mode in (
                    ("B2", "robust", "ucb"),
                    ("B4", "point", "ucb"),
                    ("B5", "robust", "mean"),
                ):
                    outputs[stage], step_metadata[stage] = candidate(
                        g_i,
                        history_basis,
                        pred,
                        samples,
                        z_samples,
                        correction_mode=correction_mode,
                        gate_mode=gate_mode,
                    )
            if random_basis is None:
                outputs["B3"] = g_i
                step_metadata["B3"] = {"conflict": False, "projected": False}
            else:
                random_coordinates = random_basis.T @ g_i
                pred, samples, z_samples = measure(
                    model, microbatches, random_basis, random_coordinates
                )
                outputs["B3"], step_metadata["B3"] = candidate(
                    g_i,
                    random_basis,
                    pred,
                    samples,
                    z_samples,
                    correction_mode="robust",
                    gate_mode="ucb",
                )

            history.append(g_i.detach().clone())
            if len(history) > subspace_rank - 1:
                del history[: -(subspace_rank - 1)]
            return {stage: outputs[stage] for stage in ("B2", "B3", "B4", "B5")}

        def observe(pred_loss_before, pred_loss_after_raw, candidate_losses):
            step = self.global_optim_step + 1
            raw_delta = pred_loss_after_raw - pred_loss_before
            denominator = max(abs(pred_loss_before), 1e-12)

            def record(variant, *, delta, conflict, projected, gain):
                """Persist one variant's numbers; the log line is for humans."""
                path = f"rq3/{variant}"
                for name, number in (
                    ("pred_loss_before", pred_loss_before),
                    ("candidate_pred_loss_delta", delta),
                    ("conflict", conflict),
                    ("projected", projected),
                    ("gain_vs_raw", gain),
                    ("relative_gain", gain / denominator),
                ):
                    self.record_metric(f"{path}/{name}", number, step)

            record("B1", delta=raw_delta, conflict=0, projected=0, gain=0.0)
            self.logger.info(
                "RQ3 counterfactual step: step=%d | variant=B1 |"
                "pred_loss_before=%.9g | candidate_pred_loss_delta=%.9g | "
                "conflict=0 | projected=0 | gain_vs_raw=0 | relative_gain=0",
                step, pred_loss_before, raw_delta,
            )
            for stage in ("B2", "B3", "B4", "B5"):
                candidate_loss = candidate_losses.get(stage)
                if candidate_loss is None:
                    continue
                candidate_delta = candidate_loss - pred_loss_before
                gain = pred_loss_after_raw - candidate_loss
                metadata = step_metadata[stage]
                record(
                    stage,
                    delta=candidate_delta,
                    conflict=int(metadata["conflict"]),
                    projected=int(metadata["projected"]),
                    gain=gain,
                )
                self.logger.info(
                    "RQ3 counterfactual step: step=%d | variant=%s | "
                    "pred_loss_before=%.9g | candidate_pred_loss_delta=%.9g | "
                    "conflict=%d | projected=%d | gain_vs_raw=%.9g | "
                    "relative_gain=%.9g",
                    step,
                    stage,
                    pred_loss_before,
                    candidate_delta,
                    int(metadata["conflict"]),
                    int(metadata["projected"]),
                    gain,
                    gain / denominator,
                )

        self.logger.info(
            "RQ3 shared-trajectory analysis: realised_update=B1 | "
            "counterfactual_variants=B2,B3,B4,B5 | rank=%d | epsilon=%g | "
            "confidence_delta=%g",
            subspace_rank,
            epsilon,
            confidence_delta,
        )
        return self.fit(
            data_loader,
            phase_name,
            num_epoch,
            accum_steps=accum_steps,
            on_epoch_end=on_epoch_end,
            counterfactual_update_generator=generate,
            counterfactual_step_observer=observe,
        )

    def fit_rq4_counterfactual(
        self,
        data_loader,
        phase_name,
        num_epoch,
        *,
        base_rank,
        base_epsilon,
        cone_delta,
        ranks,
        epsilons,
        gate_deltas,
        accum_steps=None,
        on_epoch_end=None,
    ):
        """Evaluate RQ4 parameter sweeps on one realised B1 trajectory."""
        ranks = list(dict.fromkeys(ranks))
        epsilons = list(dict.fromkeys(epsilons))
        gate_deltas = list(dict.fromkeys(gate_deltas))
        if any(rank < 2 for rank in ranks):
            raise ValueError("all RQ4 ranks must be at least 2")
        if any(epsilon <= 0 for epsilon in epsilons):
            raise ValueError("all RQ4 epsilons must be positive")
        if any(not 0.0 < delta < 1.0 for delta in gate_deltas):
            raise ValueError("all RQ4 gate deltas must lie in (0, 1)")
        if base_rank < 2 or base_epsilon <= 0 or not 0.0 < cone_delta < 1.0:
            raise ValueError("invalid RQ4 base parameters")
        if accum_steps is None:
            accum_steps = self.accum_steps
        prediction_gradient_loader = data_loader.get(
            "train_eval", data_loader["train"]
        )

        max_rank = max([base_rank, *ranks])
        history: list[torch.Tensor] = []
        step_metadata: dict[str, dict[str, bool]] = {}

        def measure(model, microbatches, basis, update_coordinates, epsilon):
            coordinate_batches = []
            diagnostic_batches = []
            for rng_index, raw_batch in enumerate(prediction_gradient_loader):
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                    model._move_batch_to_device(raw_batch)
                )
                coordinates, _, diagnostics = (
                    model._compute_subspace_prediction_gradient(
                        X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                        epsilon=epsilon,
                        basis=basis,
                        update_coordinates=update_coordinates,
                        rng_index=rng_index,
                    )
                )
                coordinate_batches.append(coordinates)
                if diagnostics:
                    diagnostic_batches.append(diagnostics)
            if diagnostic_batches:
                coordinate_samples = torch.cat([
                    item["coordinate_samples"] for item in diagnostic_batches
                ])
                prediction_coordinates = coordinate_samples.mean(dim=0)
            else:
                prediction_coordinates = torch.stack(coordinate_batches).mean(dim=0)
                coordinate_samples = prediction_coordinates.unsqueeze(0)
            return prediction_coordinates, coordinate_samples

        def build_candidate(
            g_i,
            basis,
            prediction_coordinates,
            coordinate_samples,
            gate_delta,
        ):
            update_coordinates = basis.T @ g_i
            z_samples = coordinate_samples @ update_coordinates
            upper = self._student_t_upper_confidence_bound(z_samples, gate_delta)
            conflict = upper.item() < 0.0
            project = (
                conflict
                and torch.linalg.vector_norm(prediction_coordinates).item() > 1e-12
            )
            candidate_coordinates = update_coordinates
            if project:
                candidate_coordinates, _ = self._robust_subspace_projection(
                    update_coordinates,
                    coordinate_samples,
                    cone_delta,
                )
            update = g_i + basis @ (candidate_coordinates - update_coordinates)
            return update if project else g_i, {
                "conflict": conflict,
                "projected": project,
            }

        def generate(model, microbatches, raw_adamw_update):
            nonlocal step_metadata
            g_i = raw_adamw_update
            basis_cache = {}
            measurement_cache = {}
            candidate_cache = {}
            outputs = {}
            step_metadata = {}

            def evaluate_variant(name, rank, epsilon, gate_delta):
                basis = basis_cache.get(rank)
                if rank not in basis_cache:
                    basis = model._orthonormal_update_basis(
                        [g_i, *reversed(history)], rank
                    )
                    basis_cache[rank] = basis
                if basis is None or basis.shape[1] < rank:
                    outputs[name] = g_i
                    step_metadata[name] = {"conflict": False, "projected": False}
                    return
                measurement_key = (rank, epsilon)
                if measurement_key not in measurement_cache:
                    update_coordinates = basis.T @ g_i
                    measurement_cache[measurement_key] = measure(
                        model,
                        microbatches,
                        basis,
                        update_coordinates,
                        epsilon,
                    )
                candidate_key = (rank, epsilon, gate_delta)
                if candidate_key not in candidate_cache:
                    pred, samples = measurement_cache[measurement_key]
                    candidate_cache[candidate_key] = build_candidate(
                        g_i, basis, pred, samples, gate_delta
                    )
                outputs[name], step_metadata[name] = candidate_cache[candidate_key]

            for rank in ranks:
                model._synchronize_for_timing()
                rank_started = time.perf_counter()
                evaluate_variant(f"rank={rank}", rank, base_epsilon, cone_delta)
                model._synchronize_for_timing()
                rank_seconds = time.perf_counter() - rank_started
                self.logger.info(
                    "RQ4 TIP timing: epoch=%d | step=%d | rank=%d | "
                    "wall_seconds=%.9f",
                    self.global_epoch + 1,
                    self.global_optim_step + 1,
                    rank,
                    rank_seconds,
                )
                # One rank per key: several ranks share a step, so the rank
                # belongs in the metric path or the last one would win.
                self.record_metric(
                    f"timing/tip_seconds/rank{rank}",
                    rank_seconds,
                    self.global_optim_step + 1,
                )
            for epsilon in epsilons:
                evaluate_variant(
                    f"epsilon={epsilon:.12g}",
                    base_rank,
                    epsilon,
                    cone_delta,
                )
            for delta in gate_deltas:
                evaluate_variant(
                    f"gate_delta={delta:.12g}",
                    base_rank,
                    base_epsilon,
                    delta,
                )

            history.append(g_i.detach().clone())
            if len(history) > max_rank - 1:
                del history[: -(max_rank - 1)]
            return outputs

        def observe(pred_loss_before, pred_loss_after_raw, candidate_losses):
            step = self.global_optim_step + 1
            denominator = max(abs(pred_loss_before), 1e-12)
            raw_delta = pred_loss_after_raw - pred_loss_before

            def record(sweep, value, *, delta, conflict, projected, gain):
                """Persist one candidate's numbers; the log line is for humans."""
                path = f"rq4/{sweep}/{value}"
                for name, number in (
                    ("pred_loss_before", pred_loss_before),
                    ("candidate_pred_loss_delta", delta),
                    ("conflict", conflict),
                    ("projected", projected),
                    ("gain_vs_raw", gain),
                    ("relative_gain", gain / denominator),
                ):
                    self.record_metric(f"{path}/{name}", number, step)

            record("B1", "base", delta=raw_delta, conflict=0, projected=0, gain=0.0)
            self.logger.info(
                "RQ4 counterfactual step: step=%d | sweep=B1 | value=base | "
                "pred_loss_before=%.9g | candidate_pred_loss_delta=%.9g | "
                "conflict=0 | projected=0 | gain_vs_raw=0 | relative_gain=0",
                step, pred_loss_before, raw_delta,
            )
            for name, candidate_loss in candidate_losses.items():
                if candidate_loss is None:
                    continue
                sweep, value = name.split("=", 1)
                candidate_delta = candidate_loss - pred_loss_before
                gain = pred_loss_after_raw - candidate_loss
                metadata = step_metadata[name]
                record(
                    sweep,
                    value,
                    delta=candidate_delta,
                    conflict=int(metadata["conflict"]),
                    projected=int(metadata["projected"]),
                    gain=gain,
                )
                self.logger.info(
                    "RQ4 counterfactual step: step=%d | sweep=%s | value=%s | "
                    "pred_loss_before=%.9g | candidate_pred_loss_delta=%.9g | "
                    "conflict=%d | projected=%d | gain_vs_raw=%.9g | "
                    "relative_gain=%.9g",
                    step,
                    sweep,
                    value,
                    pred_loss_before,
                    candidate_delta,
                    int(metadata["conflict"]),
                    int(metadata["projected"]),
                    gain,
                    gain / denominator,
                )

        self.logger.info(
            "RQ4 shared-trajectory sensitivity: realised_update=B1 | "
            "ranks=%s | epsilons=%s | gate_deltas=%s | base_rank=%d | "
            "base_epsilon=%g | cone_delta=%g",
            ranks,
            epsilons,
            gate_deltas,
            base_rank,
            base_epsilon,
            cone_delta,
        )
        return self.fit(
            data_loader,
            phase_name,
            num_epoch,
            accum_steps=accum_steps,
            on_epoch_end=on_epoch_end,
            counterfactual_update_generator=generate,
            counterfactual_step_observer=observe,
        )

    def fit_tip(
        self, data_loader, phase_name, num_epoch, subspace_rank=4,
        epsilon=1e-3, confidence_delta=0.05, accum_steps=None,
        on_epoch_end=None, basis_mode="history", correction_mode="robust",
        gate_mode="ucb", anchor_delta=0.0,
    ):
        if subspace_rank < 2:
            raise ValueError("subspace_rank must be at least 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must lie in (0, 1)")
        if anchor_delta < 0.0:
            raise ValueError("anchor_delta must be non-negative")
        if basis_mode not in {"history", "random"}:
            raise ValueError("basis_mode must be 'history' or 'random'")
        if correction_mode not in {"robust", "point"}:
            raise ValueError("correction_mode must be 'robust' or 'point'")
        if gate_mode not in {"ucb", "mean"}:
            raise ValueError("gate_mode must be 'ucb' or 'mean'")
        if accum_steps is None:
            accum_steps = self.accum_steps
        prediction_gradient_loader = data_loader.get(
            "train_eval", data_loader["train"]
        )
        self.logger.info(
            "TIP local subspace: target_rank=%d | "
            "finite_difference_directions_per_step=%d | "
            "gradient_scope=full_training_loader | "
            "step_monitor_full_loader_passes=3 | "
            "epsilon=%g | confidence_delta=%g | basis_mode=%s | "
            "correction_mode=%s | gate_mode=%s | "
            "update_space=adamw_parameter_displacement | "
            "prediction_rng=common_random_numbers",
            subspace_rank, 2 * subspace_rank,
            epsilon, confidence_delta, basis_mode, correction_mode, gate_mode,
        )

        history: list[torch.Tensor] = []
        stats = {
            "steps": 0,
            "anchor_applied": 0.0,
            "anchor_requested": 0.0,
            "anchor_backtrack": 0,
            "raw": 0,
            "ucb_conflict": 0,
            "projected": 0,
            "ucb": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "z_sem": 0.0,
            "adamw_update_norm": 0.0,
            "gp_subspace_norm": 0.0,
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
            "projected_gain_ratio": 0.0,
            "projected_gain_ratio_sq": 0.0,
        }
        last_step_projected = False

        def modifier(model, microbatches, raw_adamw_update):
            nonlocal last_step_projected
            # TIP operates on the parameter-space direction proposed by the
            # configured optimizer, after any momentum, scaling, and decay.
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
            if basis is None:
                last_step_projected = False
                stats["steps"] += 1
                return g_i
            effective_rank = basis.shape[1]
            update_coordinates = basis.T @ g_i

            coordinate_batches = []
            z_batches = []
            diagnostic_batches = []
            for rng_index, raw_batch in enumerate(prediction_gradient_loader):
                X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask = (
                    model._move_batch_to_device(raw_batch)
                )
                coordinates, z_samples, diagnostics = (
                    model._compute_subspace_prediction_gradient(
                        X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
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
            if project and anchor_delta > 0.0 and gp_norm.item() > 1e-12:
                # Task anchor.  The minimum-change projection lands exactly on
                # the boundary g_P^T d = 0, so it can only stop the forecast
                # loss from rising -- it can never push it down.  A step of
                # fixed magnitude along the forecast-gradient direction makes
                # the first-order change strictly negative, and because the
                # magnitude does not scale with the current proposal it
                # survives as the proposal shrinks (the projection alone does
                # not: it is positively homogeneous in the proposal).
                direction = prediction_coordinates / gp_norm
                pre_anchor = candidate_coordinates.clone()
                realized = anchor_delta
                for _ in range(6):
                    anchored = candidate_coordinates + realized * direction
                    if (
                        correction_mode != "robust"
                        or model._robust_cone_margin(
                            anchored, coordinate_samples, confidence_delta
                        )
                        >= 0.0
                    ):
                        candidate_coordinates = anchored
                        break
                    realized *= 0.5  # backtrack to stay inside the cone
                    stats["anchor_backtrack"] += 1
                stats["anchor_requested"] += float(anchor_delta)
                stats["anchor_applied"] += float(
                    (candidate_coordinates - pre_anchor).norm().item()
                )
            candidate = g_i + basis @ (
                candidate_coordinates - update_coordinates
            )
            correction = candidate - g_i
            correction_ratio = (
                torch.linalg.vector_norm(correction) / gi_norm.clamp_min(1e-12)
            )

            stats["steps"] += 1
            stats["raw"] += int(raw_conflict)
            stats["ucb_conflict"] += int(gate_conflict)
            stats["projected"] += int(project)
            stats["ucb"] += upper.item()
            stats["z_mean"] += z_samples.mean().item()
            z_std = z_samples.std(unbiased=z_samples.numel() > 1)
            stats["z_std"] += z_std.item()
            stats["z_sem"] += (z_std / math.sqrt(z_samples.numel())).item()
            stats["adamw_update_norm"] += gi_norm.item()
            stats["gp_subspace_norm"] += gp_norm.item()
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
            history.append(
                candidate.detach().clone() if project else g_i.detach().clone()
            )
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
                projection_gain_ratio = (
                    (raw_adamw_delta - tip_adamw_delta)
                    / max(abs(raw_adamw_delta), 1e-12)
                )
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
                prediction_change_stats[
                    "projected_gain_ratio"
                ] += projection_gain_ratio
                prediction_change_stats[
                    "projected_gain_ratio_sq"
                ] += projection_gain_ratio ** 2

        def epoch_end(epoch, train_loss, model):
            steps = stats["steps"]
            if steps:
                projected = max(stats["projected"], 1)
                averages = {
                    key: value / steps
                    for key, value in stats.items()
                    if key not in {
                        "steps", "raw", "ucb_conflict", "projected"
                    }
                }
                model.logger.info(
                    "TIP subspace: raw_conflict_rate=%.4f | "
                    "ucb_conflict_rate=%.4f | projection_rate=%.4f | "
                    "steps=%d",
                    stats["raw"] / steps,
                    stats["ucb_conflict"] / steps,
                    stats["projected"] / steps,
                    steps,
                )
                model.logger.info(
                    "TIP confidence: mean_ucb=%.6g | z_mean=%.6g | "
                    "z_std=%.6g | z_sem=%.6g | coordinate_variation=%.6g | "
                    "q_center=%.6g",
                    averages["ucb"], averages["z_mean"], averages["z_std"],
                    averages["z_sem"], averages["coordinate_variation"],
                    averages["q_center"],
                )
                anchor_requests = float(stats.get("anchor_requested", 0.0))
                model.logger.info(
                    "TIP anchor: delta=%.6g | fired_requests=%d | "
                    "requested_norm=%.6g | applied_norm=%.6g | "
                    "applied_ratio=%.6g | backtracks=%d",
                    anchor_delta,
                    int(round(anchor_requests / anchor_delta))
                    if anchor_delta > 0 else 0,
                    anchor_requests,
                    float(stats.get("anchor_applied", 0.0)),
                    float(stats.get("anchor_applied", 0.0)) / anchor_requests
                    if anchor_requests > 0 else float("nan"),
                    int(stats.get("anchor_backtrack", 0)),
                )
                model.logger.info(
                    "TIP geometry: adamw_update_norm=%.6g | "
                    "gp_subspace_norm=%.6g | "
                    "cosine=%.6g | "
                    "candidate_correction_ratio=%.6g | "
                    "applied_correction_ratio=%.6g | robust_margin=%.6g | "
                    "robust_solver_success=%.4f | robust_fallback=%.4f",
                    averages["adamw_update_norm"],
                    averages["gp_subspace_norm"],
                    averages["cosine"],
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
                projected_count = prediction_change_stats["projected_observed"]
                gain_ratio_mean = (
                    prediction_change_stats["projected_gain_ratio"]
                    / projected_count
                    if projected_count
                    else float("nan")
                )
                if projected_count > 1:
                    gain_ratio_variance = max(
                        0.0,
                        (
                            prediction_change_stats["projected_gain_ratio_sq"]
                            - projected_count * gain_ratio_mean ** 2
                        )
                        / (projected_count - 1),
                    )
                    gain_ratio_std = math.sqrt(gain_ratio_variance)
                elif projected_count == 1:
                    gain_ratio_std = 0.0
                else:
                    gain_ratio_std = float("nan")
                prediction_change_values[
                    "projected_tip_improvement_rate"
                ] = (
                    prediction_change_stats["projected_tip_adamw_nonincrease"]
                    / projected_count
                    if projected_count
                    else float("nan")
                )
                prediction_change_values[
                    "projected_tip_relative_gain_mean"
                ] = gain_ratio_mean
                prediction_change_values[
                    "projected_tip_relative_gain_std"
                ] = gain_ratio_std
                model.logger.info(
                    "TIP raw optimizer proposal: pred_loss_before=%.6g | "
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
                    "TIP projected optimizer step: pred_loss_after=%.6g | "
                    "pred_loss_delta=%.6g | pred_loss_relative_delta=%.6g | "
                    "pred_loss_nonincrease_rate=%.4f | "
                    "projected_pred_loss_delta=%.6g | "
                    "projected_pred_loss_relative_delta=%.6g | "
                    "projected_pred_loss_nonincrease_rate=%.4f | "
                    "tip_projection_gain=%.6g | "
                    "projected_tip_projection_gain=%.6g | "
                    "projected_tip_improvement_rate=%.4f | "
                    "projected_tip_relative_gain_mean=%.6g | "
                    "projected_tip_relative_gain_std=%.6g | "
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
                    prediction_change_values[
                        "projected_tip_improvement_rate"
                    ],
                    prediction_change_values[
                        "projected_tip_relative_gain_mean"
                    ],
                    prediction_change_values[
                        "projected_tip_relative_gain_std"
                    ],
                    prediction_change_stats["observed"],
                    prediction_change_stats["projected_observed"],
                )
                tb_values = {
                    "raw_conflict_rate": stats["raw"] / steps,
                    "ucb_conflict_rate": stats["ucb_conflict"] / steps,
                    "projection_rate": stats["projected"] / steps,
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
                    model.record_metric(f"tip/{phase_name}/{name}", value, epoch)
            for key in stats:
                stats[key] = 0 if key in {
                    "steps", "raw", "ucb_conflict", "projected"
                } else 0.0
            for key in prediction_change_stats:
                prediction_change_stats[key] = 0 if key in {
                    "observed", "raw_adamw_nonincrease",
                    "tip_adamw_nonincrease", "projected_observed",
                    "projected_raw_adamw_nonincrease",
                    "projected_tip_adamw_nonincrease",
                } else 0.0
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, model)

        return self.fit(
            data_loader, phase_name, num_epoch, accum_steps=accum_steps,
            on_epoch_end=epoch_end, optimizer_update_modifier=modifier,
            optimizer_step_observer=observe_optimizer_step,
        )
