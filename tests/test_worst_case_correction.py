#!/usr/bin/env python3
"""Numerical self-test for the ray-restricted worst-case correction.

Run directly (no pytest needed)::

    .venv/bin/python tests/test_worst_case_correction.py

It checks the two properties the design promises, on synthetic coordinate
samples whose regime is controlled by construction:

1. **Worst-case feasibility.**  Whenever the correction reports success, the
   returned coordinate vector ``x`` really does satisfy
   ``min_{c in E} c^T x >= 0`` over the ellipsoid, verified independently by
   sampling points on the ellipsoid boundary in addition to the closed-form
   margin.
2. **Beyond the point projection.**  The correction is never smaller than the
   mean-only (point) projection of the raw proposal, and the point projection
   itself is not worst-case feasible, i.e. the extra step is what absorbs the
   estimation error.

It also cross-checks the closed-form roots against the concave margin, so a
sign or algebra regression fails loudly instead of silently degrading.
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

TOL = 1e-7

# Correction magnitude relative to the proposal, collected across cases.
AMPLIFICATIONS: list[float] = []


def build_problem(rng, rank, sample_count, t_vec, conflict, noise_scale=1.0):
    """Coordinate samples whose vector t-statistic is roughly ``t_vec``."""
    direction = rng.standard_normal(rank)
    direction /= np.linalg.norm(direction)
    if direction[0] < 0:
        direction = -direction
    per_coordinate_sd = noise_scale / math.sqrt(rank)
    standard_error = per_coordinate_sd / math.sqrt(sample_count)
    mean = t_vec * standard_error * direction
    samples = mean + per_coordinate_sd * rng.standard_normal(
        (sample_count, rank)
    )
    update = np.zeros(rank)
    update[0] = -abs(conflict)
    return (
        torch.tensor(update, dtype=torch.float64),
        torch.tensor(samples, dtype=torch.float64),
    )


def analytic_margin(update, samples, delta, radius_kind, value):
    mean, covariance, beta = JointModel._worst_case_stats(
        samples, delta, radius_kind
    )
    point = torch.as_tensor(value, dtype=torch.float64)
    variance = float(torch.clamp(point @ covariance @ point, min=0.0))
    return float(mean @ point - beta * math.sqrt(variance))


def sampled_worst_case(update, samples, delta, radius_kind, value, draws, rng):
    """Monte-Carlo lower bound on min_{c in E} c^T value, on the boundary."""
    mean, covariance, beta = JointModel._worst_case_stats(
        samples, delta, radius_kind
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
    point = torch.as_tensor(value, dtype=torch.float64)
    unit = torch.tensor(rng.standard_normal((draws, mean.numel())))
    unit = unit / torch.linalg.vector_norm(unit, dim=1, keepdim=True)
    worst = mean - beta * (unit @ root.T)
    return float((worst @ point).min())


def point_projection(update, samples):
    mean = samples.mean(dim=0)
    denominator = float(mean @ mean)
    if denominator <= 1e-24:
        return update.clone()
    coefficient = float(update @ mean) / denominator
    return update - coefficient * mean


def run_case(rng, rank, sample_count, t_vec, conflict, delta, radius_kind):
    update, samples = build_problem(rng, rank, sample_count, t_vec, conflict)
    gate_margin = analytic_margin(update, samples, delta, radius_kind, update)
    if gate_margin >= 0.0:
        return None  # gate would not fire; nothing to correct
    solution, diagnostics = JointModel._worst_case_correction(
        update, samples, delta, radius_kind
    )
    solution = solution.to(dtype=torch.float64)
    return {
        "solution": solution,
        "diagnostics": diagnostics,
        "update": update,
        "samples": samples,
        "gate_margin": gate_margin,
    }


def check_feasibility(case, delta, radius_kind, rng):
    solution = case["solution"]
    diagnostics = case["diagnostics"]
    margin = analytic_margin(
        case["update"], case["samples"], delta, radius_kind, solution
    )
    sampled = sampled_worst_case(
        case["update"], case["samples"], delta, radius_kind, solution, 64, rng
    )
    if diagnostics["fallback"] == 0.0:
        assert margin >= -1e-6, f"closed-form margin {margin} < 0 with success"
        assert sampled >= -1e-6, f"sampled worst case {sampled} < 0 with success"
        assert abs(diagnostics["margin"] - margin) <= 1e-6 * max(
            1.0, abs(margin)
        ), "reported margin disagrees with the recomputed one"
    return margin, sampled


def check_beyond_point_projection(case, delta, radius_kind):
    solution = case["solution"]
    update = case["update"]
    samples = case["samples"]
    diagnostics = case["diagnostics"]
    if diagnostics["fallback"] != 0.0 or diagnostics["regime"] == 3.0:
        return None
    mean = samples.mean(dim=0)
    raw_alignment = float(update @ mean)
    # The correction must leave the mean alignment non-negative; the point
    # projection leaves it at exactly zero, so this is the "overshoot".
    corrected_alignment = float(solution @ mean)
    assert corrected_alignment >= -1e-9, (
        f"mean alignment {corrected_alignment} stayed negative"
    )
    # The imputation-descent cap: the corrected displacement must still be a
    # descent direction for the reconstruction objective, a^T x >= 0.
    descent = float(update @ solution)
    assert descent >= -1e-7, (
        f"corrected step is not a reconstruction descent step ({descent} < 0)"
    )
    # Amplification budget, enforced on the outer-root branch where the
    # correction is allowed to grow to the far end of the interval.
    amplification = float(torch.linalg.vector_norm(solution - update)) / max(
        float(torch.linalg.vector_norm(update)), 1e-12
    )
    assert amplification <= 1.0 + 1e-6, (
        f"correction amplifies the proposal by {amplification}x"
    )
    AMPLIFICATIONS.append(amplification)
    if raw_alignment >= 0.0:
        # No mean conflict to project away: the magnitude comparison against
        # the point projection is not meaningful here.
        return None
    projection = point_projection(update, samples)
    corrected = float(torch.linalg.vector_norm(solution - update))
    baseline = float(torch.linalg.vector_norm(projection - update))
    projection_margin = analytic_margin(
        update, samples, delta, radius_kind, projection
    )
    assert corrected >= baseline - 1e-9, (
        f"correction {corrected} smaller than the point projection {baseline}"
    )
    assert projection_margin <= diagnostics["margin"] + 1e-9, (
        "point projection is not worse than the worst-case correction"
    )
    return corrected / baseline if baseline > 0 else float("nan")


def check_closed_form_roots(case, delta, radius_kind):
    """The reported interval endpoints must be roots of the concave margin."""
    diagnostics = case["diagnostics"]
    if diagnostics["fallback"] != 0.0 or diagnostics["regime"] not in (1.0, 2.0):
        return None
    mean, covariance, beta = JointModel._worst_case_stats(
        case["samples"], delta, radius_kind
    )
    update = case["update"]
    direction = mean

    def margin(value):
        point = update + value * direction
        variance = float(torch.clamp(point @ covariance @ point, min=0.0))
        return float(mean @ point - beta * math.sqrt(variance))

    scale = max(abs(margin(0.0)), 1e-12)
    for name in ("lambda1", "lambda2"):
        lam = diagnostics[name]
        if lam < 0.0:  # sentinel for "+infinity" / "not applicable"
            continue
        residual = margin(lam)
        assert abs(residual) <= 1e-6 * scale + 1e-9, (
            f"{name}={lam} is not a root of the margin (residual {residual})"
        )
    return None


def main() -> int:
    rng = np.random.default_rng(20260922)
    AMPLIFICATIONS.clear()
    report: dict[str, dict[str, int]] = {}
    by_setting: dict[tuple[str, float], dict[str, int]] = {}
    ratios: list[float] = []
    failures = 0

    cases = 0
    for radius_kind in ("chi2", "hotelling", "student_t"):
        for delta in (0.01, 0.05, 0.1, 0.3):
            for rank in (2, 3, 8):
                for sample_count in (24, 248):
                    for t_vec in (0.5, 1.0, 2.0, 4.0):
                        for conflict in (0.2, 0.35, 0.6):
                            for _ in range(3):
                                case = run_case(
                                    rng, rank, sample_count, t_vec, conflict,
                                    delta, radius_kind,
                                )
                                if case is None:
                                    continue
                                cases += 1
                                diagnostics = case["diagnostics"]
                                if diagnostics["regime"] == 1.0:
                                    regime = "bounded"
                                elif diagnostics["regime"] == 2.0:
                                    regime = "ray"
                                elif diagnostics["regime"] == 3.0:
                                    regime = "cone-fallback"
                                else:
                                    regime = "raw-ok"
                                bucket = report.setdefault(
                                    regime, {"count": 0, "feasible": 0}
                                )
                                bucket["count"] += 1
                                setting = by_setting.setdefault(
                                    (radius_kind, delta),
                                    {"bounded": 0, "ray": 0,
                                     "cone-fallback": 0, "raw-ok": 0},
                                )
                                setting[regime] += 1
                                try:
                                    margin, sampled = check_feasibility(
                                        case, delta, radius_kind, rng
                                    )
                                    bucket["feasible"] += 1
                                    ratio = check_beyond_point_projection(
                                        case, delta, radius_kind
                                    )
                                    if ratio is not None:
                                        ratios.append(ratio)
                                    check_closed_form_roots(
                                        case, delta, radius_kind
                                    )
                                except AssertionError as error:
                                    failures += 1
                                    print(f"FAIL: {error}")

    print(f"conflicting configurations exercised: {cases}")
    print(f"{'regime':>10} {'count':>8} {'feasible':>9}")
    for regime, bucket in sorted(report.items()):
        print(f"{regime:>10} {bucket['count']:>8} {bucket['feasible']:>9}")
    print()
    print(f"{'radius':>10} {'delta':>6} {'bounded':>8} {'ray':>6} "
          f"{'cone':>6} {'raw-ok':>7}")
    for (radius_kind, delta), counts in sorted(by_setting.items()):
        print(f"{radius_kind:>10} {delta:>6.2f} {counts['bounded']:>8} "
              f"{counts['ray']:>6} {counts['cone-fallback']:>6} "
              f"{counts['raw-ok']:>7}")
    if ratios:
        finite = [r for r in ratios if math.isfinite(r)]
        print(
            "correction / point-projection magnitude: "
            f"min={min(finite):.4f} median={float(np.median(finite)):.4f} "
            f"max={max(finite):.4f}"
        )
    if AMPLIFICATIONS:
        print(
            "correction / proposal magnitude: "
            f"max={max(AMPLIFICATIONS):.4f} "
            f"median={float(np.median(AMPLIFICATIONS)):.4f}"
        )
    print(f"assertion failures: {failures}")

    if failures:
        return 1
    # A handful of boundary cases sit within the tolerance of the gate, so a
    # few "already feasible" reports are expected; they must stay marginal.
    raw_ok = report.get("raw-ok", {}).get("count", 0)
    assert raw_ok <= 0.05 * max(cases, 1), (
        f"too many already-feasible raw steps ({raw_ok}/{cases})"
    )
    corrected = (
        report.get("bounded", {}).get("count", 0)
        + report.get("ray", {}).get("count", 0)
        + report.get("cone-fallback", {}).get("count", 0)
    )
    assert corrected > 0, "no conflicting configuration produced a correction"
    assert not ratios or min(ratios) >= 1.0 - 1e-9, (
        "a correction was smaller than the point projection"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
