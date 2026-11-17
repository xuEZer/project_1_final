"""PILOT displacements and the B1–B6 component comparisons."""

import torch

from src.measurement import dot
from src.models.jointModel import JointModel
from src.rng import derive_seed

METHODS = {"raw", "pilot", "random", "point", "mean_gate", "forecast_direction"}


def probe_budget(population, batch_size, current_size, probe_batches):
    """Cap the requested probe batches at the residual population size."""
    if type(probe_batches) is not int or probe_batches < 1:
        raise ValueError("probe_batches must be a positive integer")
    if batch_size < 1 or not 0 < current_size <= population:
        raise ValueError("Invalid population or optimizer batch size")
    return probe_batches, min(probe_batches * batch_size, population - current_size)


def sample_probe(population, current, count, seed, step):
    """Sample uniformly without replacement outside the optimizer batch."""
    available = torch.ones(population, dtype=torch.bool)
    available[current.cpu()] = False
    indices = available.nonzero().flatten()
    generator = torch.Generator().manual_seed(derive_seed(seed, "probe_windows", step))
    return indices[torch.randperm(len(indices), generator=generator)[:count]]


def retain_history(history, applied, capacity):
    """Retain the most recent applied displacements, including zero steps."""
    if capacity < 1:
        return []
    return [*history, applied.detach().clone()][-capacity:]


def halfspace(raw, gradient):
    """Project a displacement onto nonnegative estimated-gradient alignment."""
    output = raw.double().clone()
    alignment, norm_squared = dot(raw, gradient), dot(gradient, gradient)
    if alignment < 0 and norm_squared > 0:
        output -= alignment / norm_squared * gradient.double()
    return output.to(raw)


def update_basis(raw, history, rank, method, seed, step):
    """Build a candidate-containing basis from history or random directions."""
    if method == "random":
        generator = torch.Generator(device=raw.device).manual_seed(
            derive_seed(seed, "random_basis", step)
        )
        directions = [
            torch.randn(
                raw.shape, dtype=raw.dtype, device=raw.device, generator=generator
            )
            for _ in range(rank - 1)
        ]
    else:
        directions = list(reversed(history[-(rank - 1) :])) if rank > 1 else []
    return JointModel._orthonormal_update_basis([raw, *directions], rank)


@torch.no_grad()
def candidate(
    engine, weights, raw, history, ids, settings, method, batch_size, step, seed
):
    """Return the applied descent displacement at the current parameter state.

    B1 is reconstruction-only. B2 uses robust release and projection. B3 uses
    random lateral directions; B4 uses point projection; B5 uses mean release;
    B6 uses the estimated forecast-gradient direction after a failed release.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown update method: {method}")
    rank = int(settings["subspace_rank"])
    population = len(engine.splits["train"]["X_N"])
    batches, count = probe_budget(
        population, batch_size, len(ids), settings["probe_batches"]
    )
    info = {
        "requested_rank": rank,
        "effective_rank": 0,
        "requested_probe_batches": batches,
        "probe_windows": 0,
        "protected": 0,
        "released": 1,
        "corrected": 0,
        "reason": "raw",
    }
    if method == "raw":
        return raw, info
    basis = update_basis(raw, history, rank, method, seed, step)
    info["effective_rank"] = 0 if basis is None else basis.shape[1]
    if basis is None or basis.shape[1] < rank:
        info["reason"] = "insufficient_rank"
        return raw, info
    if count < 2:
        info["reason"] = "insufficient_probe"
        return raw, info
    probe = sample_probe(population, ids, count, seed, step)
    queried = torch.cat([ids.cpu(), probe])
    samples = engine.coordinates(queried, basis, float(settings["epsilon"]), step)
    mean, covariance, beta, _ = JointModel._finite_population_ellipsoid(
        samples[: len(ids)], samples[len(ids) :], population, float(settings["delta"])
    )
    coordinates = basis.T @ raw.double()
    margin = JointModel._ellipsoid_robust_margin(coordinates, mean, covariance, beta)
    release = dot(coordinates, mean) >= 0 if method == "mean_gate" else margin >= 0
    info.update(
        probe_windows=count,
        protected=1,
        released=int(release),
        raw_robust_margin=margin,
    )
    if release:
        info["reason"] = "released"
        return raw, info
    if method == "point":
        corrected = halfspace(coordinates, mean)
    elif method == "forecast_direction":
        corrected = mean.clone()
    else:
        corrected, _ = JointModel._robust_cone_solve(
            coordinates, mean, covariance, beta
        )
    corrected_norm = corrected.norm()
    if corrected_norm > 1e-12:
        corrected *= coordinates.norm() / corrected_norm
    else:
        corrected.zero_()
    result = (basis @ corrected).to(raw)
    if not torch.isfinite(result).all():
        raise ArithmeticError("Nonfinite PILOT displacement")
    info.update(
        corrected=1,
        reason="corrected",
        applied_robust_margin=JointModel._ellipsoid_robust_margin(
            basis.T @ result.double(), mean, covariance, beta
        ),
    )
    return result, info
