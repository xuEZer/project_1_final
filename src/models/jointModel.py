"""Trainable imputation, frozen forecasting, and PILOT measurements."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from torch import nn

from src.models.pilot_geometry import minimum_change_projection
from src.rng import torch_seed_scope


class JointModel(nn.Module):
    """Pair a trainable imputer with an inference-only forecaster."""

    def __init__(self, imputer: nn.Module, predictor: nn.Module, device: str):
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor
        self.device = device
        self.base_seed = 0
        self.imputer.requires_grad_(True)
        self.predictor.requires_grad_(False)
        self.predictor.eval()

    def configure_base_seed(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def train(self, mode: bool = True):
        super().train(mode)
        self.predictor.eval()
        return self

    def _synchronize_for_timing(self) -> None:
        torch.cuda.synchronize(self.device)

    @staticmethod
    def _capture_training_modes(module: nn.Module) -> dict[nn.Module, bool]:
        return {submodule: submodule.training for submodule in module.modules()}

    @staticmethod
    def _restore_training_modes(training_modes: dict[nn.Module, bool]) -> None:
        for module, is_training in training_modes.items():
            module.training = is_training

    @staticmethod
    def _orthonormal_update_basis(
        vectors: list[torch.Tensor], max_rank: int, relative_tolerance: float = 1e-6
    ) -> torch.Tensor | None:
        """Orthonormalize in order, placing the candidate displacement first."""
        axes = []
        for vector in vectors:
            residual = vector.detach().double().clone()
            original_norm = residual.norm()
            if original_norm <= 1e-12:
                continue
            for _ in range(2):
                for axis in axes:
                    residual.sub_(axis * torch.dot(axis, residual))
            residual_norm = residual.norm()
            if residual_norm / original_norm > relative_tolerance:
                axes.append(residual / residual_norm)
            if len(axes) == max_rank:
                break
        return torch.stack(axes, dim=1) if axes else None

    @torch.no_grad()
    def prediction_coordinates(
        self,
        inputs,
        missing,
        targets,
        target_missing,
        basis,
        epsilon,
        step,
        chunk_index,
    ):
        """Measure per-window central differences using paired query randomness."""
        parameters = list(self.imputer.parameters())
        original = [parameter.detach().clone() for parameter in parameters]
        saved_buffers = {
            name: value.detach().clone() for name, value in self.imputer.named_buffers()
        }
        modes = self._capture_training_modes(self.imputer)
        observed = 1 - target_missing
        counts = observed.sum((1, 2))
        if torch.any(counts == 0):
            raise ValueError(
                "Every queried window must have an observed forecast target"
            )

        def perturb(direction, scale):
            offset = 0
            for parameter, value in zip(parameters, original, strict=True):
                count = parameter.numel()
                parameter.copy_(value).add_(
                    direction[offset : offset + count].view_as(parameter), alpha=scale
                )
                offset += count
            for name, buffer in self.imputer.named_buffers():
                buffer.copy_(saved_buffers[name])

        columns = []
        try:
            self.imputer.eval()
            for axis in range(basis.shape[1]):
                losses = []
                for sign in (1, -1):
                    perturb(basis[:, axis], sign * epsilon)
                    with torch_seed_scope(
                        self.base_seed, "finite_difference", step, chunk_index, axis
                    ):
                        completed = (
                            torch.cat(
                                [
                                    self.imputer(x, mask)["imp_data"]
                                    for x, mask in zip(
                                        inputs.split(1), missing.split(1), strict=True
                                    )
                                ]
                            )
                            if getattr(self.imputer, "windowwise_pilot_queries", False)
                            else self.imputer(inputs, missing)["imp_data"]
                        )
                        prediction = self.predictor(completed)["pred_data"]
                    losses.append(
                        (
                            (prediction.double() - targets.double()).square() * observed
                        ).sum((1, 2))
                        / counts
                    )
                columns.append((losses[0] - losses[1]) / (2 * epsilon))
        finally:
            for parameter, value in zip(parameters, original, strict=True):
                parameter.copy_(value)
            for name, buffer in self.imputer.named_buffers():
                buffer.copy_(saved_buffers[name])
            self._restore_training_modes(modes)
        return torch.stack(columns, dim=1)

    @staticmethod
    def _ellipsoid_robust_margin(
        coordinates: torch.Tensor,
        mean: torch.Tensor,
        covariance_mean: torch.Tensor,
        beta: float,
    ) -> float:
        """Robust lower alignment of ``coordinates`` for a given ellipsoid.

        This is the safe-release statistic: the raw coordinate is released only
        when it is non-negative.
        """
        value = coordinates.detach().to(dtype=torch.float64)
        variance = torch.clamp(value @ covariance_mean @ value, min=0.0)
        return float(mean @ value - beta * torch.sqrt(variance).item())

    @staticmethod
    def _ellipsoid_radius(
        rank: int,
        sample_count: int,
        delta: float,
    ) -> float:
        """Algorithm 1's Hotelling radius, or its chi-square fallback for m <= r."""
        if sample_count <= rank:
            return float(np.sqrt(chi2_dist.ppf(1.0 - delta, rank)))
        scale = rank * (sample_count - 1) / (sample_count - rank)
        return float(
            np.sqrt(scale * f_dist.ppf(1.0 - delta, rank, sample_count - rank))
        )

    @staticmethod
    def _finite_population_ellipsoid(
        batch_coordinates: torch.Tensor,
        probe_coordinates: torch.Tensor,
        population_size: int,
        delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
        """Window-level finite-population estimator, covariance and radius.

        ``batch_coordinates`` holds the per-window finite-difference coordinates of the windows in
        the current optimizer batch and ``probe_coordinates`` those of the
        probe windows, which were drawn uniformly without replacement from the
        complement of that batch.  Conditioning on the optimizer batch makes
        the un-queried residual windows a fixed finite population, so the
        full-training mean is estimated by

            c_hat = (b/N) mean(batch) + (M/N) mean(probe),   M = N - b,

        which is conditionally unbiased, with finite-population covariance

            (M/N)^2 (1 - f)/m S_probe,   f = m/M,

        estimated without bias by the probe sample covariance when ``m >= 2``.
        The ellipsoid targets the finite-difference mean. Approximation bias
        relative to the true derivative is bounded separately in the theorem.
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
        if (
            batch_windows == 0
            or probe_windows < 2
            or batch_windows + probe_windows > population_size
        ):
            raise ValueError(
                "Expected a nonempty batch and at least two disjoint probe windows"
            )
        total = float(population_size)
        residual = total - batch_windows
        fraction = probe_windows / residual

        probe_mean = probe.mean(dim=0)
        estimate = (batch_windows / total) * batch.mean(dim=0) + (
            residual / total
        ) * probe_mean
        centered = probe - probe_mean
        sample_covariance = centered.T @ centered / (probe_windows - 1.0)
        covariance = (
            (residual / total) ** 2
            * (1.0 - fraction)
            / max(probe_windows, 1.0)
            * sample_covariance
        )
        ridge = max(float(torch.trace(covariance).item()) / max(rank, 1) * 1e-8, 1e-12)
        covariance = covariance + ridge * torch.eye(
            rank, dtype=torch.float64, device=device
        )
        beta = JointModel._ellipsoid_radius(rank, len(probe), delta)
        return (
            estimate,
            covariance,
            beta,
            {
                "population_windows": total,
                "batch_windows": batch_windows,
                "probe_windows": probe_windows,
                "residual_windows": residual,
                "sampling_fraction": fraction,
                "beta": beta,
            },
        )

    @staticmethod
    def _robust_cone_solve(
        raw: torch.Tensor,
        mean: torch.Tensor,
        covariance_mean: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Project in the small coordinate space; transfer only rank-sized arrays."""
        array = minimum_change_projection(
            raw.detach().double().cpu().numpy(),
            mean.detach().double().cpu().numpy(),
            covariance_mean.detach().double().cpu().numpy(),
            beta,
        )
        solution = torch.as_tensor(array, dtype=raw.dtype, device=raw.device)
        margin = JointModel._ellipsoid_robust_margin(
            solution, mean, covariance_mean, beta
        )
        return solution, {"margin": margin}
