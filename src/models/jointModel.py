"""Joint training wrapper for imputation and prediction models.

The imputer is trained via its reconstruction / imputation loss.  The
predictor is **inference-only** — it runs inside ``torch.no_grad()``,
does not produce a loss, and never participates in backpropagation.
"""

from __future__ import annotations

import collections.abc
import logging
import math
import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from scipy.stats import t as student_t
from torch import nn
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

    # Radius families for the covariance-weighted coordinate ellipsoid.
    CONE_RADIUS_KINDS = ("hotelling", "chi2", "student_t")

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
    def compute_val_prediction_loss(
        self,
        val_loader: DataLoader,
        *,
        rng_namespace: str | None = None,
        rng_step: int | None = None,
    ) -> float:
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
                for batch_index, batch in enumerate(val_loader):
                    (
                        X_N, X_NA, X_NA_mask, X_A_mask, Y_N, Y_N_mask,
                    ) = self._move_batch_to_device(batch)
                    inputs = (
                        X_NA, X_NA_mask, X_N, X_A_mask, Y_N, Y_N_mask,
                    )
                    if rng_namespace is None:
                        output = self.forward(*inputs, run_predictor=True)
                    else:
                        with torch_seed_scope(
                            self.base_seed,
                            rng_namespace,
                            self.global_optim_step
                            if rng_step is None else rng_step,
                            batch_index,
                        ):
                            output = self.forward(*inputs, run_predictor=True)
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
                coordinate_samples.append((q_plus - q_minus) / (2.0 * epsilon))
        finally:
            for p, base in zip(params, theta, strict=True):
                p.copy_(base)
            self._restore_training_modes(training_modes)

        per_sample_coordinates = torch.stack(coordinate_samples, dim=1)
        gradient_coordinates = per_sample_coordinates.mean(dim=0)
        z_samples = per_sample_coordinates @ update_coordinates
        diagnostics = {
            "coordinate_samples": per_sample_coordinates,
            # Mean over one probe batch: the covariance cone consumes these
            # as unit-level samples, whose between-batch spread honestly
            # reflects dependence inside a batch.
            "unit_coordinate_samples": per_sample_coordinates.mean(dim=0),
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

    @staticmethod
    def _ellipsoid_radius(
        rank: int,
        sample_count: int,
        delta: float,
        radius_kind: str = "hotelling",
    ) -> float:
        """Radius of a rank-dimensional confidence ellipsoid at level 1-delta.

        ``hotelling`` is the exact normal-theory radius
        ``sqrt(rank (n-1)/(n-rank) F_{rank,n-rank,1-delta})``, ``chi2`` its
        large-sample limit, and ``student_t`` the univariate quantile of the
        original implementation (the rank = 1 special case).  The radius of a
        rank-dimensional ellipsoid must grow like ``sqrt(rank)``, which the
        Student-t choice does not.
        """
        if radius_kind == "student_t":
            return float(student_t.ppf(1.0 - delta, df=max(sample_count - 1, 1)))
        if radius_kind == "hotelling":
            degrees = max(sample_count - 1, 1)
            if degrees - rank + 1 <= 0:
                return JointModel._ellipsoid_radius(
                    rank, sample_count, delta, "chi2"
                )
            scale = rank * degrees / (degrees - rank + 1)
            return float(
                np.sqrt(scale * f_dist.ppf(1.0 - delta, rank, degrees - rank + 1))
            )
        if radius_kind != "chi2":
            raise ValueError(
                f"radius_kind must be one of {JointModel.CONE_RADIUS_KINDS}, "
                f"got {radius_kind!r}"
            )
        return float(np.sqrt(chi2_dist.ppf(1.0 - delta, df=max(rank, 1))))

    @staticmethod
    def _finite_population_ellipsoid(
        batch_coordinates: torch.Tensor,
        probe_coordinates: torch.Tensor,
        population_size: int,
        delta: float,
        radius_kind: str = "hotelling",
        epsilon: float = 0.0,
        third_derivative_bound: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
        """Window-level finite-population estimator, covariance and radius.

        Implements the revised statistical analysis.  ``batch_coordinates``
        holds the per-window finite-difference coordinates of the windows in
        the current optimizer batch and ``probe_coordinates`` those of the
        probe windows, which were drawn uniformly without replacement from the
        complement of that batch.  Conditioning on the optimizer batch makes
        the un-queried residual windows a fixed finite population, so the
        full-training mean is estimated by

            c_hat = (b/N) mean(batch) + (M/N) mean(probe),   M = N - b,

        which is conditionally unbiased, with finite-population covariance

            (M/N)^2 (1 - f)/m S_probe,   f = m/M,

        estimated without bias by the probe sample covariance when ``m >= 2``.
        The radius is inflated by the deterministic central-difference bound
        ``sqrt(rank) M_3 epsilon^2 / 6`` measured in the ellipsoid's own metric,
        ``beta_eff = beta + rho_eps / sqrt(lambda_min(Sigma))``.
        """
        batch = batch_coordinates.detach().to(dtype=torch.float64)
        probe = probe_coordinates.detach().to(dtype=torch.float64)
        if batch.ndim != 2 or probe.ndim != 2:
            raise ValueError("coordinate blocks must have shape [n, rank]")
        if batch.shape[1] != probe.shape[1]:
            raise ValueError("batch and probe blocks must share the rank")
        rank = batch.shape[1]
        device = probe.device
        batch_windows = float(batch.shape[0])
        probe_windows = float(probe.shape[0])
        total = float(max(int(population_size), int(batch_windows + probe_windows)))
        residual = max(total - batch_windows, 1.0)
        fraction = min(probe_windows / residual, 1.0)

        probe_mean = probe.mean(dim=0)
        estimate = (batch_windows / total) * batch.mean(dim=0) + (
            residual / total
        ) * probe_mean
        if probe.shape[0] >= 2:
            centered = probe - probe_mean
            sample_covariance = centered.T @ centered / (probe_windows - 1.0)
        else:
            sample_covariance = torch.zeros(
                (rank, rank), dtype=torch.float64, device=device
            )
        covariance = (
            (residual / total) ** 2
            * (1.0 - fraction)
            / max(probe_windows, 1.0)
            * sample_covariance
        )
        ridge = max(
            float(torch.trace(covariance).item()) / max(rank, 1) * 1e-8, 1e-12
        )
        covariance = covariance + ridge * torch.eye(
            rank, dtype=torch.float64, device=device
        )
        beta = JointModel._ellipsoid_radius(
            rank, max(int(probe_windows), 2), delta, radius_kind
        )
        if third_derivative_bound > 0.0 and epsilon > 0.0:
            smallest = float(
                torch.linalg.eigvalsh(covariance)[0].clamp_min(1e-300).item()
            )
            beta += (
                np.sqrt(rank)
                * third_derivative_bound
                * epsilon**2
                / 6.0
                / np.sqrt(smallest)
            )
        return estimate, covariance, beta, {
            "population_windows": total,
            "batch_windows": batch_windows,
            "probe_windows": probe_windows,
            "residual_windows": residual,
            "sampling_fraction": fraction,
            "beta": beta,
        }

    @staticmethod
    def _robust_cone_solve(
        raw: torch.Tensor,
        mean: torch.Tensor,
        covariance_mean: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Min-change coordinate inside the robust cone, by KKT.

        Solves ``min ||x - raw||^2`` subject to
        ``mean^T x - beta sqrt(x^T Cov x) >= 0``.  The KKT system of the
        equivalent quadratic cone is diagonalised, which reduces the constrained
        problem to a monotone one-dimensional root and avoids a per-step SciPy
        optimiser and its Python callbacks.  A statistically unidentified cone
        falls back to the zero coordinate vector, which safely skips the
        conflicting step.
        """
        work_device = raw.device

        def robust_margin(value: torch.Tensor) -> torch.Tensor:
            variance = torch.clamp(value @ covariance_mean @ value, min=0.0)
            return mean @ value - beta * torch.sqrt(variance)

        raw_margin = robust_margin(raw)
        if raw_margin >= -1e-12:
            solution = raw
            success = 1.0
        else:
            # Squaring the active boundary gives x^T M x = 0 with
            # M = beta^2 Sigma - mean mean^T.  KKT then yields
            # x(lambda) = (I + lambda M)^-1 raw.  In M's eigenbasis the
            # boundary equation is scalar and monotone on the PSD interval.
            cone_matrix = (
                beta * beta * covariance_mean - torch.outer(mean, mean)
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(cone_matrix)
            negative = eigenvalues < -1e-14
            solution = torch.zeros_like(raw)
            success = 0.0
            if torch.any(negative):
                rotated_raw = eigenvectors.T @ raw
                most_negative = eigenvalues[0]
                pole = -1.0 / most_negative

                def boundary_value(multiplier: torch.Tensor) -> torch.Tensor:
                    denominator = 1.0 + multiplier * eigenvalues
                    return torch.sum(
                        eigenvalues * rotated_raw.square()
                        / denominator.square()
                    )

                def bisect_root(
                    low: torch.Tensor,
                    high: torch.Tensor,
                ) -> torch.Tensor:
                    low_value = boundary_value(low)
                    for _ in range(80):
                        middle = 0.5 * (low + high)
                        middle_value = boundary_value(middle)
                        if low_value * middle_value <= 0.0:
                            high = middle
                        else:
                            low = middle
                            low_value = middle_value
                    return 0.5 * (low + high)

                # The squared boundary has two nappes separated by the pole
                # of (I + lambda M)^-1.  Consider a root on each side, retain
                # only the requested (positive-mean) nappe, then choose the
                # closest feasible point.  The cone apex is always a valid
                # candidate.
                root_candidates: list[torch.Tensor] = []
                zero = torch.zeros((), dtype=torch.float64, device=work_device)
                before_pole = pole * (1.0 - 1e-12)
                if boundary_value(zero) * boundary_value(before_pole) < 0.0:
                    root_candidates.append(bisect_root(zero, before_pole))
                after_pole = pole * (1.0 + 1e-12)
                far_after_pole = pole * 1e12
                if (
                    boundary_value(after_pole)
                    * boundary_value(far_after_pole)
                    < 0.0
                ):
                    root_candidates.append(
                        bisect_root(after_pole, far_after_pole)
                    )

                feasible = [torch.zeros_like(raw)]
                for multiplier in root_candidates:
                    candidate = eigenvectors @ (
                        rotated_raw / (1.0 + multiplier * eigenvalues)
                    )
                    if (
                        mean @ candidate >= 0.0
                        and robust_margin(candidate) >= -1e-7
                    ):
                        feasible.append(candidate)
                solution = min(
                    feasible,
                    key=lambda value: float(
                        torch.linalg.vector_norm(value - raw).item()
                    ),
                )
                success = float(len(feasible) > 1)

        if not success:
            solution.zero_()
        fallback = float(torch.linalg.vector_norm(solution).item() <= 1e-10)
        return solution, {
            "success": success,
            "margin": float(robust_margin(solution).item()),
            "fallback": fallback,
        }

    def _sync_first_moment_to_applied_update(
        self,
        parameters: list[torch.nn.Parameter],
        parameters_before: list[torch.Tensor],
        applied_update: torch.Tensor,
    ) -> None:
        """Make Adam's first moment consistent with the displacement actually applied.

        PILOT replaces the optimizer's proposed displacement *after*
        ``optimizer.step()`` has already advanced the moment estimates using the
        uncorrected imputation gradient.  The state therefore keeps implying the
        raw step, so the next proposal re-injects the component PILOT just removed,
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
        prediction_probe_batches: int = 1,
        applied_update_observer: collections.abc.Callable[[torch.Tensor], None] | None = None,
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

        Returns:
            Final training loss from the last epoch in this phase.
        """
        if accum_steps is None:
            accum_steps = getattr(self, "accum_steps", 1)
        if prediction_probe_batches < 0:
            raise ValueError("prediction_probe_batches must be non-negative")

        train_loader = data_loader["train"]
        final_train_loss = float("inf")
        self.phase_epoch = 0
        measure_wall_clock = phase_name == "rq4"

        while self.phase_epoch < num_epoch:
            epoch_number = self.global_epoch + 1
            if measure_wall_clock:
                self._synchronize_for_timing()
                epoch_started = time.perf_counter()
            total_loss = 0.0
            # PILOT estimates its prediction gradient from the current optimizer
            # batch plus one distinct, deterministically sampled probe batch.
            # Materialising the (small) epoch keeps the probe on the exact same
            # shuffled partition as the realised training trajectory without
            # advancing a second DataLoader RNG stream. B1 keeps streaming the
            # loader normally.
            uses_prediction_gradient = optimizer_update_modifier is not None
            epoch_batches = (
                list(train_loader) if uses_prediction_gradient else train_loader
            )
            num_batches = len(epoch_batches)
            num_optim_steps = 0

            self.imputer.train()

            accum_loss = 0.0
            accum_count = 0
            accum_target = 1
            accum_microbatches: list[dict[str, torch.Tensor]] = []

            for batch_idx, batch in enumerate(epoch_batches):
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
                    if measure_wall_clock:
                        self._synchronize_for_timing()

                    prediction_gradient_microbatches = accum_microbatches
                    if (
                        uses_prediction_gradient
                        and prediction_probe_batches > 0
                        and num_batches > accum_count
                    ):
                        current_start = batch_idx - accum_count + 1
                        current_indices = set(range(current_start, batch_idx + 1))
                        probe_candidates = [
                            index
                            for index in range(num_batches)
                            if index not in current_indices
                        ]
                        selected_probe_indices = []
                        remaining_candidates = list(probe_candidates)
                        for probe_ordinal in range(min(
                            prediction_probe_batches,
                            len(remaining_candidates),
                        )):
                            probe_choice = derive_seed(
                                self.base_seed,
                                "tip_probe_batch",
                                self.global_optim_step,
                                probe_ordinal,
                            ) % len(remaining_candidates)
                            selected_probe_indices.append(
                                remaining_candidates.pop(probe_choice)
                            )
                        prediction_gradient_microbatches = [
                            dict(microbatch, is_probe=False)
                            for microbatch in accum_microbatches
                        ]
                        for probe_index in selected_probe_indices:
                            (
                                probe_X_N,
                                probe_X_NA,
                                probe_X_NA_mask,
                                probe_X_A_mask,
                                probe_Y_N,
                                probe_Y_N_mask,
                            ) = self._move_batch_to_device(
                                epoch_batches[probe_index]
                            )
                            prediction_gradient_microbatches.append({
                                "X_N": probe_X_N,
                                "X_NA": probe_X_NA,
                                "X_NA_mask": probe_X_NA_mask,
                                "X_A_mask": probe_X_A_mask,
                                "Y_N": probe_Y_N,
                                "Y_N_mask": probe_Y_N_mask,
                                "is_probe": True,
                            })
                        self.logger.info(
                            "PILOT probe batch: epoch=%d | step=%d | "
                            "num_batches=%d | accum_count=%d | "
                            "current_batch=%d | probe_batches=%s | "
                            "candidate_count=%d",
                            epoch_number,
                            self.global_optim_step + 1,
                            num_batches,
                            accum_count,
                            batch_idx,
                            selected_probe_indices,
                            len(probe_candidates),
                        )

                    if optimizer_update_modifier is not None:
                        parameters = list(self.imputer.parameters())
                        parameters_before = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
                        # Advance the optimizer exactly once, including any
                        # state and weight decay, then extract its true parameter-space
                        # proposal before PILOT changes the applied displacement.
                        self.optimizer.step()
                        parameters_after_raw_adamw = [
                            parameter.detach().clone()
                            for parameter in parameters
                        ]
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
                            self,
                            prediction_gradient_microbatches,
                            raw_adamw_update,
                        )
                        offset = 0
                        with torch.no_grad():
                            if torch.equal(final_update, raw_adamw_update):
                                # Preserve the optimizer's exact floating-point state
                                # when PILOT is a no-op; reconstructing it as
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
                                "PILOT update size does not match imputer parameters"
                            )
                        modified_update = not torch.equal(
                            final_update, raw_adamw_update
                        )
                        if modified_update:
                            self._sync_first_moment_to_applied_update(
                                parameters, parameters_before, final_update
                            )
                    else:
                        parameters_before = (
                            [parameter.detach().clone() for parameter in self.imputer.parameters()]
                            if applied_update_observer is not None else None
                        )
                        self.optimizer.step()
                        if parameters_before is not None:
                            applied_update_observer(torch.cat([
                                (before - parameter.detach()).flatten()
                                for before, parameter in zip(
                                    parameters_before,
                                    self.imputer.parameters(),
                                    strict=True,
                                )
                            ]))
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

    def fit_pilot(
        self, data_loader, phase_name, num_epoch, subspace_rank=4,
        epsilon=1e-3, accum_steps=None,
        on_epoch_end=None, basis_mode="history", correction_mode="robust",
        gate_mode="safe_release", gate_delta=0.3, cone_delta=None,
        cone_radius="hotelling", finite_difference_third_derivative=0.0,
        preserve_norm=False, prediction_probe_batches=1,
        initial_history=None,
    ):
        if cone_delta is None:
            cone_delta = gate_delta
        if subspace_rank < 2:
            raise ValueError("subspace_rank must be at least 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < gate_delta < 1.0:
            raise ValueError("gate_delta must lie in (0, 1)")
        if not 0.0 < cone_delta < 1.0:
            raise ValueError("cone_delta must lie in (0, 1)")
        if prediction_probe_batches < 0:
            raise ValueError("prediction_probe_batches must be non-negative")
        if basis_mode not in {"history", "random"}:
            raise ValueError("basis_mode must be 'history' or 'random'")
        if correction_mode not in {"robust", "point", "gradient"}:
            raise ValueError(
                "correction_mode must be 'robust', 'point', or 'gradient'"
            )
        if gate_mode not in {"safe_release", "mean"}:
            raise ValueError("gate_mode must be 'safe_release' or 'mean'")
        if cone_radius not in JointModel.CONE_RADIUS_KINDS:
            raise ValueError(
                f"cone_radius must be one of {JointModel.CONE_RADIUS_KINDS}"
            )
        if finite_difference_third_derivative < 0.0:
            raise ValueError(
                "finite_difference_third_derivative must be non-negative"
            )
        self.cone_radius_kind = cone_radius
        if accum_steps is None:
            accum_steps = self.accum_steps
        self.logger.info(
            "PILOT local subspace: target_rank=%d | "
            "finite_difference_directions_per_step=%d | "
            "gradient_scope=current_optimizer_batch_plus_%d_probe_batches | "
            "epsilon=%g | delta=%g | cone_radius=%s | "
            "preserve_norm=%s | basis_mode=%s | "
            "correction_mode=%s | gate_mode=%s | "
            "update_space=adamw_parameter_displacement | "
            "prediction_rng=common_random_numbers",
            subspace_rank, 2 * subspace_rank, prediction_probe_batches,
            epsilon, gate_delta, cone_radius, preserve_norm,
            basis_mode, correction_mode, gate_mode,
        )

        history: list[torch.Tensor] = [
            displacement.detach().clone()
            for displacement in (initial_history or [])[-(subspace_rank - 1):]
        ]
        stats = {
            "steps": 0,
            "raw": 0,
            "gate_conflict": 0,
            "projected": 0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "z_sem": 0.0,
            "z_unit_mean": 0.0,
            "z_unit_sem": 0.0,
            "unit_count": 0.0,
            "adamw_update_norm": 0.0,
            "gp_subspace_norm": 0.0,
            "cosine": 0.0,
            "candidate_correction_ratio": 0.0,
            "applied_correction_ratio": 0.0,
            "coordinate_variation": 0.0,
            "robust_margin": 0.0,
            "robust_solver_success": 0.0,
            "robust_fallback": 0.0,
            "fd_sampling_fraction": 0.0,
            "fd_residual_windows": 0.0,
            "fd_steps": 0,
            "norm_scale": 0.0,
            "norm_rescaled": 0,
            "norm_scale_skipped": 0,
        }
        try:
            # Training-stream size for the finite-population correction of
            # the gate's margin: the probed windows are a without-replacement
            # sample of these windows.
            pilot_population_size: int | None = len(
                data_loader["train"].dataset
            )
        except (KeyError, TypeError, AttributeError):
            pilot_population_size = None

        def modifier(model, microbatches, raw_adamw_update):
            # PILOT operates on the parameter-space direction proposed by the
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
                stats["steps"] += 1
                return g_i
            effective_rank = basis.shape[1]
            update_coordinates = basis.T @ g_i

            coordinate_batches = []
            z_batches = []
            diagnostic_batches = []
            diagnostic_is_probe = []
            for rng_index, microbatch in enumerate(microbatches):
                coordinates, z_samples, diagnostics = (
                    model._compute_subspace_prediction_gradient(
                        microbatch["X_NA"],
                        microbatch["X_NA_mask"],
                        microbatch["X_N"],
                        microbatch["X_A_mask"],
                        microbatch["Y_N"],
                        microbatch["Y_N_mask"],
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
                    diagnostic_is_probe.append(
                        bool(microbatch.get("is_probe", False))
                    )

            if diagnostic_batches:
                coordinate_samples = torch.cat([
                    item["coordinate_samples"] for item in diagnostic_batches
                ])
                unit_coordinate_samples = torch.stack([
                    item["unit_coordinate_samples"]
                    for item in diagnostic_batches
                ])
                prediction_coordinates = coordinate_samples.mean(dim=0)
                z_samples = coordinate_samples @ update_coordinates
            else:
                prediction_coordinates = torch.stack(
                    coordinate_batches
                ).mean(dim=0)
                z_samples = torch.cat(z_batches)
                coordinate_samples = prediction_coordinates.unsqueeze(0)
                unit_coordinate_samples = coordinate_samples
            unit_z_samples = unit_coordinate_samples @ update_coordinates

            # Finite-population statistics of the full-training coordinate
            # mean.  The probe batches were drawn without replacement from the
            # complement of the current optimizer batch, so conditioning on
            # that batch makes the unqueried windows a fixed finite population
            # and the weighted estimator below conditionally unbiased.
            finite_population = None
            if (
                diagnostic_batches
                and pilot_population_size
                and any(diagnostic_is_probe)
                and not all(diagnostic_is_probe)
            ):
                batch_windows = torch.cat([
                    item["coordinate_samples"]
                    for item, is_probe in zip(
                        diagnostic_batches, diagnostic_is_probe, strict=True
                    )
                    if not is_probe
                ])
                probe_windows = torch.cat([
                    item["coordinate_samples"]
                    for item, is_probe in zip(
                        diagnostic_batches, diagnostic_is_probe, strict=True
                    )
                    if is_probe
                ])
                if probe_windows.shape[0] >= 2:
                    finite_population = model._finite_population_ellipsoid(
                        batch_windows,
                        probe_windows,
                        pilot_population_size,
                        cone_delta,
                        cone_radius,
                        epsilon,
                        finite_difference_third_derivative,
                    )
                    prediction_coordinates = finite_population[0].to(
                        dtype=update_coordinates.dtype,
                        device=update_coordinates.device,
                    )
                    stats["fd_sampling_fraction"] += finite_population[3][
                        "sampling_fraction"
                    ]
                    stats["fd_residual_windows"] += finite_population[3][
                        "residual_windows"
                    ]
                    stats["fd_steps"] += 1
            dot = torch.dot(update_coordinates, prediction_coordinates)
            # The gate treats the probed windows as a simple random sample of
            # the training stream: their number, not the batch partition,
            # sets the t-margin (full coverage would make sampling error
            # vanish).  The covariance cone below still consumes unit-level
            # (batch-mean) samples, whose between-batch spread honestly
            # reflects dependence inside a probe batch.
            gi_norm = torch.linalg.vector_norm(g_i)
            gp_norm = torch.linalg.vector_norm(prediction_coordinates)
            cosine = dot / (gi_norm * gp_norm).clamp_min(1e-12)
            raw_conflict = dot.item() < 0.0
            rank_ready = effective_rank >= subspace_rank
            finite_population_margin = None
            if finite_population is not None and gate_mode == "safe_release":
                # The correction keeps the cone_delta radius in
                # finite_population[2].  The release gate uses its own
                # gate_delta radius over the same estimate and covariance.
                # The finite-difference remainder is independent of delta,
                # so it cancels when replacing only the statistical radius.
                coordinate_rank = finite_population[1].shape[0]
                probe_count = max(int(probe_windows.shape[0]), 2)
                gate_beta = (
                    finite_population[2]
                    + model._ellipsoid_radius(
                        coordinate_rank, probe_count, gate_delta, cone_radius
                    )
                    - model._ellipsoid_radius(
                        coordinate_rank, probe_count, cone_delta, cone_radius
                    )
                )
                finite_population_margin = (
                    model._ellipsoid_robust_margin(
                        update_coordinates,
                        finite_population[0],
                        finite_population[1],
                        gate_beta,
                    )
                )
            if gate_mode == "safe_release":
                # Release the raw coordinate only when it is certified feasible
                # for the confidence set of the full-training mean; uncertainty
                # triggers correction rather than suppressing it.
                if finite_population_margin is not None:
                    gate_conflict = finite_population_margin < 0.0
                else:
                    gate_conflict = dot.item() < 0.0
                gate_conflict = dot.item() < 0.0
            project = rank_ready and gate_conflict and gp_norm.item() > 1e-12

            candidate_coordinates = update_coordinates
            robust_diagnostics = {
                "success": 0.0,
                "margin": 0.0,
                "fallback": 0.0,
            }
            if project:
                if correction_mode == "gradient":
                    # Direct-application control: replace the proposal with the
                    # estimated forecast-gradient direction (norm-preserved
                    # below), the "descend Q" arm theory predicts to be weaker.
                    scale = (
                        torch.linalg.vector_norm(update_coordinates)
                        / gp_norm.clamp_min(1e-12)
                        if preserve_norm
                        else 1.0
                    )
                    # The optimizer applies theta <- theta - d, so d must
                    # align with the forecast gradient to descend forecast loss.
                    candidate_coordinates = (
                        scale * prediction_coordinates
                    )
                    robust_diagnostics = {
                        "success": 1.0,
                        "margin": float(torch.dot(
                            prediction_coordinates, candidate_coordinates
                        ).item()),
                        "fallback": 0.0,
                    }
                elif correction_mode == "robust" and finite_population is not None:
                    solution, robust_diagnostics = model._robust_cone_solve(
                        update_coordinates.detach().to(dtype=torch.float64),
                        finite_population[0],
                        finite_population[1],
                        finite_population[2],
                    )
                    candidate_coordinates = solution.to(
                        dtype=update_coordinates.dtype,
                        device=update_coordinates.device,
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
            if project and preserve_norm:
                # Fairness control: restore the corrected displacement to the
                # norm of the optimizer's own proposal, so B1 and B2 take steps
                # of identical length and differ only in direction.  Without it
                # a correction that shrinks the step would be confounded with a
                # correction that changes its direction.
                #
                # Rescaling is safe for the guarantee: the robust lower
                # alignment is positively homogeneous, R(t x) = t R(x) for
                # t > 0, because both terms are degree one in x.  A feasible
                # correction therefore stays feasible at any positive scale,
                # and only its length changes.
                raw_norm = float(
                    torch.linalg.vector_norm(update_coordinates).item()
                )
                corrected_norm = float(
                    torch.linalg.vector_norm(candidate_coordinates).item()
                )
                if corrected_norm > 1e-12 and raw_norm > 0.0:
                    scale = raw_norm / corrected_norm
                    candidate_coordinates = candidate_coordinates * scale
                    stats["norm_scale"] += scale
                    stats["norm_rescaled"] += 1
                else:
                    stats["norm_scale_skipped"] += 1
            candidate = g_i + basis @ (
                candidate_coordinates - update_coordinates
            )
            correction = candidate - g_i

            stats["steps"] += 1
            stats["raw"] += int(raw_conflict)
            stats["gate_conflict"] += int(gate_conflict)
            stats["projected"] += int(project)
            stats["z_mean"] += z_samples.mean().item()
            z_std = z_samples.std(unbiased=z_samples.numel() > 1)
            stats["z_std"] += z_std.item()
            stats["z_sem"] += (z_std / math.sqrt(z_samples.numel())).item()
            stats["z_unit_mean"] += unit_z_samples.mean().item()
            correction = candidate - g_i
            correction_ratio = (
                torch.linalg.vector_norm(correction) / gi_norm.clamp_min(1e-12)
            )
            unit_std = unit_z_samples.std(
                unbiased=unit_z_samples.numel() > 1
            )
            stats["z_unit_sem"] += (
                unit_std / math.sqrt(unit_z_samples.numel())
            ).item()
            stats["unit_count"] += float(unit_z_samples.numel())
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


        def epoch_end(epoch, train_loss, model):
            steps = stats["steps"]
            if steps:
                projected = max(stats["projected"], 1)
                fd_steps = max(stats.get("fd_steps", 0), 1)
                averages = {
                    key: value / steps
                    for key, value in stats.items()
                    if key not in {
                        "steps", "raw", "gate_conflict", "projected",
                    }
                }
                model.logger.info(
                    "PILOT subspace: raw_conflict_rate=%.4f | "
                    "gate_conflict_rate=%.4f | projection_rate=%.4f | "
                    "steps=%d",
                    stats["raw"] / steps,
                    stats["gate_conflict"] / steps,
                    stats["projected"] / steps,
                    steps,
                )
                model.logger.info(
                    "PILOT confidence: z_mean=%.6g | "
                    "z_std=%.6g | z_sem=%.6g | coordinate_variation=%.6g",
                    averages["z_mean"], averages["z_std"],
                    averages["z_sem"], averages["coordinate_variation"],
                )
                fd_mean_residual = (
                    stats["fd_residual_windows"] / fd_steps
                )
                fd_mean_fraction = (
                    stats["fd_sampling_fraction"] / fd_steps
                )
                model.logger.info(
                    "PILOT finite population: training_windows=%s | "
                    "residual_windows=%.1f | sampling_fraction=%.4f | "
                    "optimizer_steps=%d | fd_steps=%d | "
                    "cone_radius=%s | gate_mode=%s",
                    pilot_population_size,
                    fd_mean_residual,
                    fd_mean_fraction,
                    steps,
                    stats.get("fd_steps", 0),
                    cone_radius,
                    gate_mode,
                )
                if preserve_norm:
                    model.logger.info(
                        "PILOT norm preservation: rescaled=%d | skipped=%d | "
                        "mean_scale=%.4f",
                        stats["norm_rescaled"],
                        stats["norm_scale_skipped"],
                        stats["norm_scale"] / max(stats["norm_rescaled"], 1),
                    )
                model.logger.info(
                    "PILOT geometry: adamw_update_norm=%.6g | "
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
                tb_values = {
                    "raw_conflict_rate": stats["raw"] / steps,
                    "gate_conflict_rate": stats["gate_conflict"] / steps,
                    "projection_rate": stats["projected"] / steps,
                    **averages,
                    "robust_margin": stats["robust_margin"] / projected,
                    "robust_solver_success": (
                        stats["robust_solver_success"] / projected
                    ),
                    "robust_fallback": stats["robust_fallback"] / projected,
                }
                for name, value in tb_values.items():
                    model.record_metric(f"pilot/{phase_name}/{name}", value, epoch)
            for key in stats:
                stats[key] = 0 if key in {
                    "steps", "raw", "gate_conflict", "projected",
                    "projection_rejected",
                } else 0.0
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, model)

        return self.fit(
            data_loader, phase_name, num_epoch, accum_steps=accum_steps,
            on_epoch_end=epoch_end, optimizer_update_modifier=modifier,
            prediction_probe_batches=prediction_probe_batches,
        )
