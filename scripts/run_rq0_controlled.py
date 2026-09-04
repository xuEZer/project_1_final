"""Run the fully controlled RQ0 mechanism-validation experiment.

The experiment uses analytic objectives, paired random seeds, and a local
coordinate subspace.  It writes seed-level observations, aggregated summaries,
a machine-readable report, and a publication-ready four-panel figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t


METHODS = (
    "Raw",
    "Full-oracle",
    "Subspace-oracle",
    "Subspace-ZO",
    "Subspace-ZO-UCB",
    "Robust-ZO-UCB",
)
DEFAULTS = {
    "d": 50,
    "rank": 4,
    "angle": 120.0,
    "gamma": 0.75,
    "epsilon": 1e-3,
    "repeats": 20,
    "sigma": 1e-3,
    "delta": 0.05,
    "eta": 0.05,
    "seeds": 200,
}


def vectors(d: int, rank: int, angle: float, gamma: float) -> tuple[np.ndarray, ...]:
    if rank >= d:
        raise ValueError(f"rank={rank} must be smaller than d={d}")
    phi = math.radians(angle)
    a = np.zeros(d)
    a[0] = 1.0
    b = np.zeros(d)
    b[0] = math.cos(phi)
    b[1] = math.sin(phi) * gamma
    b[rank] += math.sin(phi) * math.sqrt(max(0.0, 1.0 - gamma**2))
    basis = np.eye(d)[:, :rank]
    return a, b, basis


def loss_i(theta: np.ndarray, a: np.ndarray) -> float:
    return 0.5 * float(np.dot(theta - a, theta - a))


def loss_q(theta: np.ndarray, b: np.ndarray, rho: float) -> float:
    base = 0.5 * float(np.dot(theta - b, theta - b))
    nonlinear = 0.25 * rho * float(np.sum((theta + 0.5) ** 4))
    return base + nonlinear


def grad_q(theta: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    return theta - b + rho * (theta + 0.5) ** 3


def project_update(gradient: np.ndarray, constraint: np.ndarray, eta: float) -> np.ndarray:
    dot = float(np.dot(gradient, constraint))
    denom = float(np.dot(constraint, constraint))
    if dot >= 0.0 or denom <= 1e-24:
        return -eta * gradient
    corrected = gradient - dot / denom * constraint
    return -eta * corrected


def robust_projection(
    raw_coordinates: np.ndarray,
    samples: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    sample_count, rank = samples.shape
    if sample_count < 2:
        return np.zeros_like(raw_coordinates), {
            "success": 0.0,
            "margin": 0.0,
            "fallback": 1.0,
        }
    mean = samples.mean(axis=0)
    covariance_mean = np.atleast_2d(
        np.cov(samples, rowvar=False, ddof=1) / sample_count
    )
    ridge = max(float(np.trace(covariance_mean)) / rank * 1e-8, 1e-12)
    covariance_mean = covariance_mean + ridge * np.eye(rank)
    beta = float(student_t.ppf(1.0 - delta, sample_count - 1))

    def objective(value: np.ndarray) -> float:
        residual = value - raw_coordinates
        return 0.5 * float(residual @ residual)

    def margin(value: np.ndarray) -> float:
        variance = max(float(value @ covariance_mean @ value), 0.0)
        return float(mean @ value - beta * math.sqrt(variance))

    starts = [np.zeros_like(raw_coordinates), raw_coordinates.copy()]
    norm_sq = float(mean @ mean)
    if norm_sq > 1e-20:
        starts.append(
            raw_coordinates
            - float(raw_coordinates @ mean) / norm_sq * mean
        )
        starts.append(
            mean * (max(float(mean @ raw_coordinates), 0.0) + 1.0) / norm_sq
        )
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints={"type": "ineq", "fun": margin},
            options={"ftol": 1e-12, "maxiter": 200},
        )
        if (
            result.success
            and margin(result.x) >= -1e-7
            and (best is None or result.fun < best.fun)
        ):
            best = result
    if best is None:
        return np.zeros_like(raw_coordinates), {
            "success": 0.0,
            "margin": 0.0,
            "fallback": 1.0,
        }
    return best.x, {
        "success": 1.0,
        "margin": margin(best.x),
        "fallback": float(np.linalg.norm(best.x) <= 1e-10),
    }


def coordinate_samples(
    theta: np.ndarray,
    b: np.ndarray,
    basis: np.ndarray,
    epsilon: float,
    rho: float,
    sigma: float,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.empty((repeats, basis.shape[1]), dtype=np.float64)
    for rep in range(repeats):
        for axis in range(basis.shape[1]):
            direction = basis[:, axis]
            plus = loss_q(theta + epsilon * direction, b, rho)
            minus = loss_q(theta - epsilon * direction, b, rho)
            if sigma:
                plus += float(rng.normal(0.0, sigma))
                minus += float(rng.normal(0.0, sigma))
            samples[rep, axis] = (plus - minus) / (2.0 * epsilon)
    return samples


def run_case(
    *,
    scan: str,
    x_name: str,
    x_value: float,
    seed: int,
    d: int,
    rank: int,
    angle: float,
    gamma: float,
    epsilon: float,
    repeats: int,
    sigma: float,
    delta: float,
    eta: float,
    rho: float = 0.0,
) -> list[dict[str, float | int | str]]:
    theta = np.zeros(d)
    a, b, basis = vectors(d, rank, angle, gamma)
    g_i = theta - a
    g_p = grad_q(theta, b, rho)
    p_g_p = basis @ (basis.T @ g_p)
    true_dot = float(np.dot(g_i, g_p))
    subspace_dot = float(np.dot(g_i, p_g_p))
    true_conflict = true_dot < -1e-12
    raw_update = -eta * g_i

    rng = np.random.default_rng(seed)
    samples = coordinate_samples(
        theta, b, basis, epsilon, rho, sigma, repeats, rng
    )
    mean_coordinates = samples.mean(axis=0)
    estimated = basis @ mean_coordinates
    z_samples = samples @ (basis.T @ g_i)
    z_mean = float(z_samples.mean())
    if repeats >= 2:
        upper = z_mean + float(student_t.ppf(1.0 - delta, repeats - 1)) * float(
            z_samples.std(ddof=1) / math.sqrt(repeats)
        )
    else:
        upper = math.inf

    raw_dot = float(np.dot(g_i, g_p))
    full_update = project_update(g_i, g_p, eta)
    sub_oracle_update = project_update(g_i, p_g_p, eta)
    zo_project = z_mean < 0.0
    zo_update = project_update(g_i, estimated, eta) if zo_project else raw_update
    ucb_project = upper < 0.0
    ucb_update = project_update(g_i, estimated, eta) if ucb_project else raw_update
    robust_diagnostics = {"success": 0.0, "margin": 0.0, "fallback": 0.0}
    if ucb_project:
        robust_coordinates, robust_diagnostics = robust_projection(
            basis.T @ g_i,
            samples,
            delta,
        )
        robust_gradient = g_i + basis @ (robust_coordinates - basis.T @ g_i)
        robust_update = -eta * robust_gradient
    else:
        robust_update = raw_update
    updates = {
        "Raw": (raw_update, False),
        "Full-oracle": (full_update, raw_dot < 0.0),
        "Subspace-oracle": (sub_oracle_update, raw_dot < 0.0),
        "Subspace-ZO": (zo_update, zo_project),
        "Subspace-ZO-UCB": (ucb_update, ucb_project),
        "Robust-ZO-UCB": (robust_update, ucb_project),
    }

    exact_coordinates = basis.T @ g_p
    coordinate_error = float(np.linalg.norm(mean_coordinates - exact_coordinates))
    denom = float(np.linalg.norm(estimated) * np.linalg.norm(p_g_p))
    cosine = float(np.dot(estimated, p_g_p) / denom) if denom > 1e-24 else math.nan
    coverage = float(np.linalg.norm(p_g_p) / np.linalg.norm(g_p))
    base_q = loss_q(theta, b, rho)
    base_i = loss_i(theta, a)
    raw_delta_q = loss_q(theta + raw_update, b, rho) - base_q
    records = []
    for method in METHODS:
        update, projected = updates[method]
        delta_q = loss_q(theta + update, b, rho) - base_q
        delta_i = loss_i(theta + update, a) - base_i
        records.append(
            {
                "scan": scan,
                "x_name": x_name,
                "x_value": x_value,
                "seed": seed,
                "method": method,
                "d": d,
                "rank": rank,
                "angle": angle,
                "gamma": gamma,
                "coverage": coverage,
                "epsilon": epsilon,
                "repeats": repeats,
                "sigma": sigma,
                "delta": delta,
                "eta": eta,
                "rho": rho,
                "true_dot": true_dot,
                "subspace_dot": subspace_dot,
                "true_conflict": int(true_conflict),
                "estimated_dot": z_mean,
                "ucb": upper,
                "projected": int(projected),
                "correct_detection": int(projected == true_conflict),
                "false_projection": int(projected and not true_conflict),
                "missed_conflict": int((not projected) and true_conflict),
                "delta_q": delta_q,
                "delta_i": delta_i,
                "exact_nonincrease": int(delta_q <= 1e-12),
                "first_order_change": float(np.dot(g_p, update)),
                "first_order_constraint_satisfied": int(
                    float(np.dot(g_p, update)) <= 1e-12
                ),
                "delta_q_reduction_vs_raw": raw_delta_q - delta_q,
                "correction_distance": float(np.linalg.norm(update - raw_update)),
                "estimate_cosine": cosine,
                "coordinate_error": coordinate_error,
                "ucb_covers": int(upper >= true_dot),
                "robust_margin": (
                    robust_diagnostics["margin"]
                    if method == "Robust-ZO-UCB" else math.nan
                ),
                "robust_solver_success": (
                    robust_diagnostics["success"]
                    if method == "Robust-ZO-UCB" else math.nan
                ),
                "robust_fallback": (
                    robust_diagnostics["fallback"]
                    if method == "Robust-ZO-UCB" else math.nan
                ),
            }
        )
    return records


def build_cases(seeds: int) -> list[dict]:
    base = dict(DEFAULTS)
    base["seeds"] = seeds
    cases: list[dict] = []

    def add(scan: str, x_name: str, x_value: float, *, stochastic: bool = True, **overrides):
        cfg = {
            "scan": scan,
            "x_name": x_name,
            "x_value": float(x_value),
            "d": base["d"],
            "rank": base["rank"],
            "angle": base["angle"],
            "gamma": base["gamma"],
            "epsilon": base["epsilon"],
            "repeats": base["repeats"],
            "sigma": base["sigma"] if stochastic else 0.0,
            "delta": base["delta"],
            "eta": base["eta"],
            "rho": 0.0,
        }
        cfg.update(overrides)
        cfg["seed_count"] = seeds if cfg["sigma"] > 0 else 1
        cases.append(cfg)

    for angle in range(30, 151, 10):
        add("angle", "angle_deg", angle, angle=angle)
    for rank in (2, 4, 8):
        for gamma in np.linspace(0.0, 1.0, 11):
            add(
                "coverage",
                "gamma",
                gamma,
                rank=rank,
                gamma=gamma,
                stochastic=False,
            )
    for epsilon in (
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
    ):
        add(
            "epsilon",
            "epsilon",
            epsilon,
            epsilon=epsilon,
            rho=0.1,
            stochastic=False,
        )
    for eta in np.linspace(0.01, 0.10, 10):
        add("locality", "eta", eta, eta=eta, stochastic=False)
    for angle, label in ((90.0, "uncertainty_null"), (120.0, "uncertainty_power")):
        for sigma in (0.0, 1e-4, 1e-3, 1e-2):
            for repeats in (5, 10, 20, 50):
                for delta in (
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10,
                ):
                    add(
                        label,
                        "delta",
                        delta,
                        angle=angle,
                        sigma=sigma,
                        repeats=repeats,
                        delta=delta,
                    )
    return cases


def aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    keys = (
        "scan", "x_name", "x_value", "method", "rank", "angle", "gamma",
        "coverage", "epsilon", "repeats", "sigma", "delta", "eta", "rho",
    )
    metrics = (
        "true_dot", "subspace_dot", "correct_detection", "false_projection",
        "missed_conflict", "delta_q",
        "delta_i", "exact_nonincrease", "first_order_change",
        "first_order_constraint_satisfied", "delta_q_reduction_vs_raw",
        "correction_distance",
        "estimate_cosine", "coordinate_error", "ucb_covers", "projected",
        "robust_margin", "robust_solver_success", "robust_fallback",
    )
    for row in records:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for group_key, rows in grouped.items():
        item = dict(zip(keys, group_key, strict=True))
        item["n"] = len(rows)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            if len(values) > 1:
                item[f"{metric}_ci95"] = float(
                    student_t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1) / math.sqrt(len(values))
                )
            else:
                item[f"{metric}_ci95"] = 0.0
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(success_rate: float, sample_count: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a Bernoulli rate."""
    if sample_count <= 0 or not math.isfinite(success_rate):
        return math.nan, math.nan
    z = 1.959963984540054
    z_squared = z * z
    denominator = 1.0 + z_squared / sample_count
    center = (success_rate + z_squared / (2.0 * sample_count)) / denominator
    radius = (
        z
        * math.sqrt(
            success_rate * (1.0 - success_rate) / sample_count
            + z_squared / (4.0 * sample_count * sample_count)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def make_figure(summary: list[dict], path: Path) -> dict[str, float]:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.1), constrained_layout=True)
    colors = {
        "Raw": "#6B7280",
        "Full-oracle": "#111827",
        "Subspace-oracle": "#2563EB",
        "Subspace-ZO": "#F59E0B",
        "Subspace-ZO-UCB": "#DC2626",
        "Robust-ZO-UCB": "#059669",
    }
    display_names = {
        "Raw": "Raw",
        "Full-oracle": "Full oracle",
        "Subspace-oracle": "Subspace oracle",
        "Subspace-ZO": "Subspace ZO",
        "Subspace-ZO-UCB": "ZO with margin",
        "Robust-ZO-UCB": "Robust ZO with margin",
    }

    ax = axes[0, 0]
    ax.axvspan(25, 90, color="#DCFCE7", alpha=0.35, linewidth=0)
    ax.axvspan(90, 155, color="#FEE2E2", alpha=0.45, linewidth=0)
    for method in METHODS:
        rows = sorted(
            [r for r in summary if r["scan"] == "angle" and r["method"] == method],
            key=lambda r: r["x_value"],
        )
        ax.errorbar(
            [r["x_value"] for r in rows],
            [r["delta_q_mean"] for r in rows],
            yerr=[r["delta_q_ci95"] for r in rows],
            marker="o",
            label=display_names[method],
            color=colors[method],
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            markersize=4,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(90, color="black", linestyle="--", linewidth=1.0)
    ax.text(88, 0.042, r"Predicted boundary  $90^{\circ}$", ha="right", va="top", fontsize=11)
    ax.text(56, -0.039, "Aligned", ha="center", va="bottom", fontsize=11, color="#166534")
    ax.text(126, 0.041, "Conflict", ha="center", va="top", fontsize=11, color="#991B1B")
    ax.set(
        title=r"(a) Conflict begins at $90^{\circ}$",
        xlabel=r"Gradient angle $\varphi$ (degrees)",
        ylabel=r"One-step prediction change $\Delta Q$",
    )

    ax = axes[0, 1]
    pair_rows = [r for r in summary if r["method"] == "Raw"]
    unique_pairs = {}
    for row in pair_rows:
        key = (
            row["true_dot_mean"],
            row["subspace_dot_mean"],
            row["rank"],
            row["angle"],
            row["gamma"],
            row["rho"],
        )
        unique_pairs[key] = row
    pair_rows = list(unique_pairs.values())
    full_dots = np.asarray([r["true_dot_mean"] for r in pair_rows])
    subspace_dots = np.asarray([r["subspace_dot_mean"] for r in pair_rows])
    axis_limit = 1.08 * float(np.max(np.abs(np.concatenate((full_dots, subspace_dots)))))
    ax.fill_between(
        [-axis_limit, 0.0],
        [0.0, 0.0],
        [axis_limit, axis_limit],
        color="#FEE2E2",
        alpha=0.45,
        linewidth=0,
    )
    ax.fill_between(
        [0.0, axis_limit],
        [-axis_limit, -axis_limit],
        [0.0, 0.0],
        color="#FEE2E2",
        alpha=0.45,
        linewidth=0,
    )
    rank_styles = {
        2: ("#7C3AED", "o"),
        4: (colors["Subspace-oracle"], "o"),
        8: ("#0891B2", "o"),
    }
    for rank in sorted({int(r["rank"]) for r in pair_rows}):
        rows = [r for r in pair_rows if int(r["rank"]) == rank]
        color, marker = rank_styles[rank]
        ax.scatter(
            [r["true_dot_mean"] for r in rows],
            [r["subspace_dot_mean"] for r in rows],
            s=28,
            alpha=0.72,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            label=rf"$r={rank}$",
            zorder=3,
        )
    ax.plot(
        [-axis_limit, axis_limit],
        [-axis_limit, axis_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=r"$y=x$",
        zorder=2,
    )
    ax.axhline(0.0, color="#6B7280", linewidth=0.7)
    ax.axvline(0.0, color="#6B7280", linewidth=0.7)
    tolerance = 1e-12
    full_signs = np.where(full_dots < -tolerance, -1, np.where(full_dots > tolerance, 1, 0))
    subspace_signs = np.where(
        subspace_dots < -tolerance,
        -1,
        np.where(subspace_dots > tolerance, 1, 0),
    )
    sign_agreement = 100.0 * float(np.mean(full_signs == subspace_signs))
    ax.text(
        -0.54 * axis_limit,
        0.72 * axis_limit,
        "Sign mismatch",
        ha="center",
        va="center",
        fontsize=11,
        color="#991B1B",
    )
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        handlelength=1.3,
        handletextpad=0.45,
        borderaxespad=0.0,
        fontsize=7,
    )
    ax.set(
        title="(b) Subspace preserves paired evidence",
        xlabel=r"Full inner product  $g_I^\top g_P$",
        ylabel=r"Subspace inner product  $g_I^\top P_{\mathcal{S}}g_P$",
    )
    ax.yaxis.labelpad = 18

    ax = axes[0, 2]
    rows = sorted(
        [r for r in summary if r["scan"] == "epsilon" and r["method"] == "Subspace-ZO"],
        key=lambda r: r["epsilon"],
    )
    eps = np.asarray([r["epsilon"] for r in rows])
    err = np.asarray([r["coordinate_error_mean"] for r in rows])
    valid = err > 1e-14
    slope = float(np.polyfit(np.log10(eps[valid]), np.log10(err[valid]), 1)[0])
    ax.loglog(
        eps,
        err,
        marker="o",
        color=colors["Subspace-ZO"],
        label=f"Observed slope  {slope:.3f}",
    )
    ref = err[valid][0] * (eps / eps[valid][0]) ** 2
    ax.loglog(eps, ref, linestyle="--", color="black", label=r"Theory slope  $2$")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set(
        title=r"(c) Difference error follows $O(\varepsilon^2)$",
        xlabel=r"Perturbation scale $\varepsilon$",
        ylabel="Coordinate error",
    )

    ax = axes[1, 0]
    locality_slopes = {}
    for method in ("Raw", "Full-oracle"):
        rows = sorted(
            [r for r in summary if r["scan"] == "locality" and r["method"] == method],
            key=lambda r: r["eta"],
        )
        eta_values = np.asarray([r["eta"] for r in rows])
        delta_values = np.asarray([r["delta_q_mean"] for r in rows])
        method_slope = float(
            np.polyfit(np.log10(eta_values), np.log10(delta_values), 1)[0]
        )
        locality_slopes[method] = method_slope
        ax.loglog(
            eta_values,
            delta_values,
            marker="o",
            color=colors[method],
            label=f"{display_names[method]}  slope {method_slope:.2f}",
        )
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set(
        title="(d) Projection removes first order harm",
        xlabel=r"Learning rate $\eta$",
        ylabel=r"One-step prediction change $\Delta Q$",
    )

    ax = axes[1, 1]
    rows = sorted(
        [
            r for r in summary
            if r["scan"] == "uncertainty_null"
            and r["method"] == "Subspace-ZO-UCB"
            and r["sigma"] == 1e-3
            and r["repeats"] == 20
        ],
        key=lambda r: r["delta"],
    )
    rates = np.asarray([r["false_projection_mean"] for r in rows])
    intervals = [
        wilson_interval(rate, int(row["n"]))
        for rate, row in zip(rates, rows, strict=True)
    ]
    lower = np.asarray([interval[0] for interval in intervals])
    upper = np.asarray([interval[1] for interval in intervals])
    ax.errorbar(
        [r["delta"] for r in rows],
        rates,
        yerr=np.maximum(0.0, np.vstack((rates - lower, upper - rates))),
        marker="o",
        label="Empirical rate",
        color=colors["Robust-ZO-UCB"],
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        markersize=4,
    )
    ax.plot([0.01, 0.10], [0.01, 0.10], linestyle="--", color="black", label="Reference")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set(
        title="(e) Margin tracks the intervention level",
        xlabel=r"Margin parameter $\delta$",
        ylabel="Boundary intervention rate",
    )

    ax = axes[1, 2]
    angles = tuple(float(angle) for angle in range(100, 151, 10))
    robust_values = {}
    for method in ("Subspace-ZO-UCB", "Robust-ZO-UCB"):
        values = []
        cis = []
        for angle in angles:
            rows = [
                r for r in summary
                if r["scan"] == "angle"
                and r["method"] == method
                and r["angle"] == angle
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    f"Expected one angle row for {method} at {angle}, got {len(rows)}"
                )
            values.append(100.0 * rows[0]["first_order_constraint_satisfied_mean"])
            cis.append(100.0 * rows[0]["first_order_constraint_satisfied_ci95"])
        robust_values[method] = dict(zip(angles, values, strict=True))
        ax.errorbar(
            angles,
            values,
            yerr=cis,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[method],
            label=display_names[method],
        )
    ax.set_xticks(angles, [rf"${int(angle)}^\circ$" for angle in angles])
    ax.set_ylim(0, 105)
    ax.set(
        title="(f) Robust correction restores satisfaction",
        xlabel=r"Conflict angle $\varphi$",
        ylabel="Constraint satisfaction rate (%)",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=6,
        frameon=False,
        fontsize=11,
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.title.set_fontsize(16)
        axis.xaxis.label.set_fontsize(14)
        axis.yaxis.label.set_fontsize(14)
        axis.tick_params(axis="both", labelsize=12)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(11)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "epsilon_loglog_slope": slope,
        "raw_locality_loglog_slope": locality_slopes["Raw"],
        "full_oracle_locality_loglog_slope": locality_slopes["Full-oracle"],
        "coverage_conflict_sign_agreement": sign_agreement / 100.0,
        "robust_satisfaction_120": robust_values["Robust-ZO-UCB"][120.0] / 100.0,
        "robust_satisfaction_150": robust_values["Robust-ZO-UCB"][150.0] / 100.0,
    }


def findings(summary: list[dict], extras: dict[str, float]) -> dict:
    def pick(**conditions):
        rows = [
            row for row in summary
            if all(row.get(key) == value for key, value in conditions.items())
        ]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one row for {conditions}, got {len(rows)}")
        return rows[0]

    angle_rows = [
        row for row in summary
        if row["scan"] == "angle" and row["method"] == "Subspace-ZO-UCB"
    ]
    null = pick(
        scan="uncertainty_null", method="Subspace-ZO-UCB",
        sigma=0.001, repeats=20, delta=0.05,
    )
    power = pick(
        scan="uncertainty_power", method="Subspace-ZO-UCB",
        sigma=0.001, repeats=20, delta=0.05,
    )
    robust_power = pick(
        scan="uncertainty_power", method="Robust-ZO-UCB",
        sigma=0.001, repeats=20, delta=0.05,
    )
    coverage_rows = [
        row for row in summary
        if row["scan"] == "coverage" and row["rank"] == 4
        and row["method"] in {"Full-oracle", "Subspace-oracle"}
    ]
    full_distances = [
        row["correction_distance_mean"]
        for row in coverage_rows if row["method"] == "Full-oracle"
    ]
    sub_distances = [
        row["correction_distance_mean"]
        for row in coverage_rows if row["method"] == "Subspace-oracle"
    ]
    return {
        **DEFAULTS,
        **extras,
        "angle_scan_ucb_detection": {
            str(int(row["angle"])): row["correct_detection_mean"] for row in angle_rows
        },
        "null_false_projection_rate_delta_005": null["false_projection_mean"],
        "null_ucb_coverage_delta_005": null["ucb_covers_mean"],
        "conflict_detection_power_delta_005": power["projected_mean"],
        "baseline_ucb_first_order_satisfaction_delta_005": (
            power["first_order_constraint_satisfied_mean"]
        ),
        "robust_ucb_first_order_satisfaction_delta_005": (
            robust_power["first_order_constraint_satisfied_mean"]
        ),
        "baseline_ucb_delta_q_delta_005": power["delta_q_mean"],
        "robust_ucb_delta_q_delta_005": robust_power["delta_q_mean"],
        "robust_solver_success_delta_005": (
            robust_power["robust_solver_success_mean"]
        ),
        "robust_fallback_delta_005": robust_power["robust_fallback_mean"],
        "full_oracle_correction_distance_range": [
            min(full_distances), max(full_distances)
        ],
        "subspace_oracle_correction_distance_range": [
            min(sub_distances), max(sub_distances)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=DEFAULTS["seeds"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/rq0"))
    parser.add_argument(
        "--figure", type=Path, default=Path("paper/Figures/rq0_controlled.pdf")
    )
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")

    records: list[dict] = []
    cases = build_cases(args.seeds)
    for case in cases:
        seed_count = case.pop("seed_count")
        for seed in range(seed_count):
            records.extend(run_case(seed=seed, **case))

    summary = aggregate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_level.csv", records)
    write_csv(args.output_dir / "summary.csv", summary)
    extras = make_figure(summary, args.figure)
    report = findings(summary, extras)
    report["case_count"] = len(cases)
    report["seed_level_rows"] = len(records)
    report["figure"] = str(args.figure)
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
