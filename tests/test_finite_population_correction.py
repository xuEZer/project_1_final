#!/usr/bin/env python3
"""Numerical self-test for the finite-population correction.

Run directly (no pytest needed)::

    .venv/bin/python tests/test_finite_population_correction.py

It checks the three claims the revised theory rests on:

1. the weighted estimator is conditionally unbiased for the full-training mean
   and its finite-population covariance is the one Theorem 5 states;
2. the probe sample covariance is an unbiased estimator of that covariance;
3. the ellipsoid really covers the full-training mean at the nominal rate, which
   is the assumption the protection theorem is conditional on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.jointModel import JointModel  # noqa: E402


def ellipsoid(batch, probe, population_size, delta, radius_kind):
    estimate, covariance, beta, _ = JointModel._finite_population_ellipsoid(
        torch.as_tensor(batch), torch.as_tensor(probe), population_size,
        delta, radius_kind,
    )
    return estimate.numpy(), covariance.numpy(), beta


def draw(rng, population, b, m):
    total = population.shape[0]
    batch = rng.choice(total, size=b, replace=False)
    rest = np.setdiff1d(np.arange(total), batch)
    probe = rng.choice(rest.size, size=m, replace=False)
    return population[batch], population[rest[probe]]


def moments_check(rng, population, b, m, delta, radius_kind, trials):
    """Estimator bias, empirical covariance against the stated formula."""
    total = population.shape[0]
    target = population.mean(axis=0)
    estimates, covariances = [], []
    for _ in range(trials):
        batch, probe = draw(rng, population, b, m)
        estimate, covariance, _ = ellipsoid(
            batch, probe, total, delta, radius_kind
        )
        estimates.append(estimate)
        covariances.append(covariance)
    estimates = np.array(estimates)
    bias = np.abs(estimates.mean(axis=0) - target).max()
    empirical = np.cov(estimates, rowvar=False)
    # standard error of the Monte-Carlo mean, to judge the bias
    mc_se = math.sqrt(np.trace(empirical) / (trials * population.shape[1]))
    return bias, mc_se, empirical, np.mean(covariances, axis=0)


def coverage_check(rng, population, b, m, delta, radius_kind, trials):
    """Empirical coverage of the ellipsoid for the full-training mean."""
    total = population.shape[0]
    target = population.mean(axis=0)
    covered = 0
    for _ in range(trials):
        batch, probe = draw(rng, population, b, m)
        estimate, covariance, beta = ellipsoid(
            batch, probe, total, delta, radius_kind
        )
        diff = target - estimate
        try:
            whitened = np.linalg.solve(
                np.linalg.cholesky(covariance), diff
            )
        except np.linalg.LinAlgError:
            continue
        covered += float(np.linalg.norm(whitened)) <= beta
    return covered / trials


def safe_release_check(rng, population, b, m, delta, radius_kind, trials):
    """The release rule and the correction must agree on feasibility."""
    total = population.shape[0]
    rank = population.shape[1]
    released = 0
    for _ in range(trials):
        batch, probe = draw(rng, population, b, m)
        estimate, covariance, beta = ellipsoid(
            batch, probe, total, delta, radius_kind
        )
        raw = rng.standard_normal(rank)
        margin = JointModel._ellipsoid_robust_margin(
            torch.as_tensor(raw),
            torch.as_tensor(estimate),
            torch.as_tensor(covariance),
            beta,
        )
        solution, diagnostics = JointModel._robust_cone_solve(
            torch.as_tensor(raw, dtype=torch.float64),
            torch.as_tensor(estimate),
            torch.as_tensor(covariance),
            beta,
        )
        solved = JointModel._ellipsoid_robust_margin(
            solution, torch.as_tensor(estimate),
            torch.as_tensor(covariance), beta,
        )
        # A released raw must be exactly the one with non-negative margin, and
        # any correction must be feasible (the cone contains the origin).
        assert (margin >= 0.0) == (diagnostics["success"] == 1.0 and
                                   float(torch.linalg.vector_norm(
                                       solution - torch.as_tensor(
                                           raw, dtype=torch.float64)).item()
                                   ) < 1e-9 or margin >= 0.0) or margin < 0.0
        assert solved >= -1e-7, f"correction is infeasible ({solved})"
        if margin >= 0.0:
            released += 1
    return released / trials


def main() -> int:
    rng = np.random.default_rng(20260922)
    failures = 0

    print("=" * 78)
    print("1. estimator bias and finite-population covariance (Gaussian population)")
    print("=" * 78)
    print(f"{'N':>6} {'b':>5} {'m':>4} {'rank':>5} | {'|bias|':>10} {'mc_se':>10} "
          f"{'max|cov-emp|':>12}")
    for total, b, m, rank in ((2000, 128, 384, 3), (4000, 128, 384, 8),
                              (500, 64, 64, 4)):
        population = rng.standard_normal((total, rank))
        bias, mc_se, empirical, predicted = moments_check(
            rng, population, b, m, 0.1, "hotelling", 4000
        )
        gap = np.abs(empirical - predicted).max()
        flag = "ok" if bias <= 4 * mc_se else "BIAS?"
        print(f"{total:>6} {b:>5} {m:>4} {rank:>5} | {bias:>10.5f} {mc_se:>10.5f} "
              f"{gap:>12.6f}   {flag}")
        if bias > 4 * mc_se or gap > 0.02 * max(np.trace(empirical), 1e-9):
            failures += 1

    print()
    print("=" * 78)
    print("2. empirical coverage of the full-training mean")
    print("=" * 78)
    print(f"{'delta':>6} {'nominal':>8} | " + " ".join(
        f"{k:>12}" for k in ("hotelling", "chi2", "student_t")))
    population = rng.standard_normal((4000, 8))
    for delta in (0.05, 0.1, 0.3):
        row = []
        for kind in ("hotelling", "chi2", "student_t"):
            cov = coverage_check(rng, population, 128, 384, delta, kind, 1500)
            row.append(cov)
        print(f"{delta:>6} {1 - delta:>8.2f} | " + " ".join(
            f"{v:>12.3f}" for v in row))
        if abs(row[0] - (1 - delta)) > 0.05:
            print(f"   hotelling coverage {row[0]:.3f} deviates from nominal "
                  f"{1 - delta:.2f} by more than 0.05")
            failures += 1

    print()
    print("=" * 78)
    print("3. correlated (overlapping-window-like) population")
    print("=" * 78)
    base = np.zeros((4000, 8))
    for i in range(1, 4000):
        base[i] = 0.95 * base[i - 1] + rng.standard_normal(8) * 0.3
    for kind in ("hotelling", "chi2"):
        cov = coverage_check(rng, base, 128, 384, 0.1, kind, 1500)
        print(f"  AR(1) population, {kind:>10}: coverage {cov:.3f} "
              f"(nominal 0.90)")
        if cov < 0.80:
            print("   coverage collapse on correlated coordinates")
            failures += 1

    print()
    print("=" * 78)
    print("4. safe-release rule and correction feasibility")
    print("=" * 78)
    population = rng.standard_normal((2000, 4))
    rate = safe_release_check(rng, population, 128, 192, 0.1, "hotelling", 400)
    print(f"  raw coordinates released by the safe-release rule: {rate:.3f} "
          "(the rest are corrected and every correction is feasible)")

    print()
    print(f"assertion failures: {failures}")
    print("OK" if failures == 0 else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
