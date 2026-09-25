"""Analytic, paired single-step controlled experiment for PILOT.

A fixed finite population of window losses is constructed for each population
seed. All methods share the same candidate, batch, probe sample, and queries.
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

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})
import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist
from scipy.stats import t as student_t


METHODS = (
    "Raw", "Full oracle", "Subspace oracle", "Point ZO",
    "Mean gate + robust projection", "Robust gate + point projection", "PILOT",
)
DEFAULTS = dict(
    d=50, rank=3, angle=120.0, gamma=0.75, eta=0.05, epsilon=1e-3,
    rho=0.0, sigma=0.5, N=512, b=64, m=256, delta=0.1,
    population_seeds=50, probe_draws=10, preserve_norm=True,
)
METRICS = (
    "true_dot", "subspace_dot", "alignment_error", "coverage_fraction",
    "estimated_coverage_fraction",
    "coordinate_error", "fd_error", "estimation_error", "set_coverage",
    "released", "intervened", "false_release", "false_intervention",
    "residual_conflict", "delta_q", "q_final", "delta_i", "first_order_change",
    "positive_first_order_change", "effective_correction",
    "correction_distance", "update_norm", "zero_update", "solver_success",
    "solver_fallback",
)


def cases():
    base = DEFAULTS
    out = []
    def add(scan, value, x_name, stochastic=False, **changes):
        cfg = {k: base[k] for k in (
            "d", "rank", "angle", "gamma", "eta", "epsilon", "rho",
            "sigma", "N", "b", "m", "delta", "preserve_norm"
        )}
        cfg.update(changes)
        cfg["scan"], cfg["x_name"], cfg["x_value"] = scan, x_name, float(value)
        cfg["case_id"] = f"{scan}_{len(out):03d}"
        cfg["population_count"] = base["population_seeds"] if stochastic else 1
        cfg["probe_draws"] = base["probe_draws"] if stochastic else 1
        out.append(cfg)
    for phi in range(30, 151, 10):
        add("A_angle", phi, "angle_deg", True, angle=float(phi), sigma=.5)
    for gamma in np.linspace(0, 1, 11):
        add("B_coverage", gamma, "gamma", True, gamma=float(gamma), sigma=.5)
    for gamma in (0., .1, .5, 1.):
        add("B_high_dimension", gamma, "gamma", True,
            d=500, gamma=gamma, sigma=.5)
    for eps in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        add("C_difference", eps, "epsilon", True, epsilon=eps, rho=0.1, sigma=.5)
    for eta in np.linspace(.01, .10, 10):
        add("D_locality", eta, "eta", True, eta=float(eta), sigma=.5)
        add("D_full_probe", eta, "eta", True, eta=float(eta),
            m=base["N"]-base["b"], sigma=.5)
    for phi in (85.0, 95.0, 120.0):
        for m in (16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 416, 448):
            for sigma in (.1, .5):
                add("E_probe", m / (base["N"] - base["b"]), "probe_fraction",
                    True, angle=phi, m=m, sigma=sigma)
    for delta in (.05, .2):
        for m in (16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 416, 448):
            add("E_probe", m / (base["N"] - base["b"]), "probe_fraction",
                True, angle=120.0, m=m, sigma=.5, delta=delta)
    for phi in (85.0, 89.0, 90.0, 91.0, 95.0, 120.0):
        for delta in (.05, .1, .2, .3, .4, .5):
            add("F_delta", delta, "delta", True, angle=phi, delta=delta)
    for phi in (85.0, 89.0, 90.0, 91.0, 95.0):
        for m in (64, 128, 256, 384, 448):
            add("F_probe_rate", m / (base["N"]-base["b"]),
                "probe_fraction", True, angle=phi, m=m, delta=.1)
    return out


def coverage_refinement_cases():
    """Resolve the small positive lateral-coverage regime on a log grid."""
    gammas = np.geomspace(1e-4, 1., 17)
    return [dict(
        {k: DEFAULTS[k] for k in (
            "d", "rank", "angle", "gamma", "eta", "epsilon", "rho",
            "sigma", "N", "b", "m", "delta", "preserve_norm"
        )},
        gamma=float(gamma), sigma=.5,
        scan="B_coverage_log", x_name="gamma", x_value=float(gamma),
        case_id=f"B_coverage_log_{j:03d}",
        population_count=DEFAULTS["population_seeds"],
        probe_draws=DEFAULTS["probe_draws"],
    ) for j, gamma in enumerate(gammas)]


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as h:
        return [
            {k: v if k in ("case_id", "scan", "x_name", "method")
             else float(v) for k, v in row.items()}
            for row in csv.DictReader(h)
        ]


def geometry(cfg):
    d, r = cfg["d"], cfg["rank"]
    if not (d > r >= 2 and 0 <= cfg["gamma"] <= 1):
        raise ValueError("Require d > rank >= 2 and gamma in [0,1]")
    e1 = np.eye(d)[0]
    V = np.eye(d)[:, :r]
    phi = math.radians(cfg["angle"])
    q = math.cos(phi) * e1
    q[1] = math.sin(phi) * cfg["gamma"]
    q[r] = math.sin(phi) * math.sqrt(max(0, 1 - cfg["gamma"] ** 2))
    d_raw = -cfg["eta"] * e1
    return e1, V, q, d_raw


def population(cfg, pop_seed, V, q):
    rng = np.random.default_rng(710000 + pop_seed)
    xi = rng.normal(size=(cfg["N"], cfg["rank"]))
    xi -= xi.mean(axis=0, keepdims=True)
    # Fixed b_i for all positive and negative queries in this population.
    targets = q + cfg["rho"] / 8 + cfg["sigma"] * xi @ V.T
    B = rng.choice(cfg["N"], size=cfg["b"], replace=False)
    complement = np.setdiff1d(np.arange(cfg["N"]), B, assume_unique=True)
    return targets, B, complement


def coordinates(targets, indices, cfg):
    """Evaluate Q_i at theta = +/- epsilon e_j for the selected windows."""
    chosen = targets[indices]
    eps, rho, rank = cfg["epsilon"], cfg["rho"], cfg["rank"]
    base = (.5 * np.sum(chosen * chosen, axis=1, keepdims=True)
            + rho / 4 * cfg["d"] * .5 ** 4)
    plus = (base - eps * chosen[:, :rank] + .5 * eps**2
            + rho / 4 * ((.5 + eps)**4 - .5**4))
    minus = (base + eps * chosen[:, :rank] + .5 * eps**2
             + rho / 4 * ((.5 - eps)**4 - .5**4))
    return (plus - minus) / (2 * eps)


def sparse_vector(value):
    """Store a complete vector compactly using zero-based nonzero indices."""
    return json.dumps({str(j): float(v) for j, v in enumerate(value)
                       if abs(float(v)) > 1e-14}, separators=(",", ":"))


def robust_margin(x, c, cov, beta):
    return float(c @ x - beta * math.sqrt(max(0.0, x @ cov @ x)))


def point_project(a, c):
    dot = float(c @ a)
    denom = float(c @ c)
    return a if dot >= 0 or denom < 1e-25 else a - dot / denom * c


def preserve(x, a, enabled):
    norm = float(np.linalg.norm(x))
    if enabled and norm > 1e-10:
        return x * (float(np.linalg.norm(a)) / norm)
    return x


def robust_project(a, c, cov, beta):
    if robust_margin(a, c, cov, beta) >= -1e-11:
        return a.copy(), 1, 0
    # The cone has no nonzero feasible direction when this dual norm is too small.
    dual_sq = float(c @ np.linalg.solve(cov, c))
    if dual_sq <= beta * beta * (1 + 1e-10):
        return np.zeros_like(a), 1, 0
    def objective(x):
        z = x - a
        return .5 * float(z @ z)
    def jac_objective(x):
        return x - a
    def constraint(x):
        return robust_margin(x, c, cov, beta)
    def jac_constraint(x):
        norm = math.sqrt(max(float(x @ cov @ x), 0.0))
        if norm <= 1e-15:
            return c
        return c - beta * (cov @ x) / norm
    # A feasible direction is Sigma^{-1} c. The zero vector is always feasible.
    direction = np.linalg.solve(cov, c)
    direction *= float(np.linalg.norm(a)) / max(float(np.linalg.norm(direction)), 1e-15)
    result = minimize(
        objective, direction, jac=jac_objective, method="SLSQP",
        constraints={"type": "ineq", "fun": constraint, "jac": jac_constraint},
        options={"ftol": 1e-11, "maxiter": 100},
    )
    if result.success and constraint(result.x) >= -1e-8:
        return np.asarray(result.x), 1, 0
    return np.zeros_like(a), 0, 1


def loss_q(theta, targets, rho):
    diff = targets - theta
    return .5 * float(np.mean(np.einsum("ij,ij->i", diff, diff))) + (
        rho / 4 * float(np.sum((theta + .5) ** 4))
    )


def evaluate_case(cfg, pop_seed, probe_seed, targets, B, complement, e1, V, q, d_raw, q0):
    N, b, m, r = cfg["N"], cfg["b"], cfg["m"], cfg["rank"]
    rng = np.random.default_rng(920000 + 1009 * pop_seed + probe_seed)
    P = rng.choice(complement, size=m, replace=False)
    cb = coordinates(targets, B, cfg)
    cp = coordinates(targets, P, cfg)
    c = (b / N) * cb.mean(axis=0) + ((N-b) / N) * cp.mean(axis=0)
    sample_cov = np.atleast_2d(np.cov(cp, rowvar=False, ddof=1))
    fp_cov = ((N-b) / N) ** 2 * (1 - m/(N-b)) * sample_cov / m
    ridge = max(1e-8 * float(np.trace(fp_cov)) / r, 1e-12)
    cov = fp_cov + ridge * np.eye(r)
    beta = math.sqrt(r * (m-1)/(m-r) * float(f_dist.ppf(1-cfg["delta"], r, m-r)))
    a = V.T @ d_raw
    mean_target = targets.mean(axis=0)
    true_g = -mean_target + cfg["rho"] / 8
    true_c = V.T @ true_g
    true_dot = float(true_g @ d_raw)
    conflict = true_dot < -1e-12
    robust_release = robust_margin(a, c, cov, beta) >= 0
    mean_release = float(c @ a) >= 0
    full_x = d_raw if not conflict else d_raw - true_dot / float(true_g @ true_g) * true_g
    sub_x = point_project(a, true_c)
    point_x = point_project(a, c)
    robust_x = a.copy()
    success, fallback = 1, 0
    if not robust_release or not mean_release:
        robust_x, success, fallback = robust_project(a, c, cov, beta)
    methods = {
        "Raw": (d_raw, False, 1, 0),
        "Full oracle": (full_x, conflict, 1, 0),
        "Subspace oracle": (V @ sub_x, conflict, 1, 0),
        "Point ZO": (V @ point_x, not mean_release, 1, 0),
        "Mean gate + robust projection": (
            V @ (robust_x if not mean_release else a), not mean_release,
            success if not mean_release else 1, fallback if not mean_release else 0,
        ),
        "Robust gate + point projection": (
            V @ (point_x if not robust_release else a), not robust_release, 1, 0,
        ),
        "PILOT": (
            V @ (robust_x if not robust_release else a), not robust_release,
            success if not robust_release else 1, fallback if not robust_release else 0,
        ),
    }
    # Population-only diagnostics are evaluated after every decision is fixed.
    true_fd = coordinates(targets, np.arange(N), cfg).mean(axis=0)
    covered = float(
        (true_fd-c) @ np.linalg.solve(cov, true_fd-c) <= beta**2 + 1e-10
    )
    i0 = .5
    coord_error = float(np.linalg.norm(c-true_c))
    fd_error = float(np.linalg.norm(true_fd-true_c))
    est_error = float(np.linalg.norm(c-true_fd))
    coverage_fraction = float(np.linalg.norm(true_c)/np.linalg.norm(true_g))
    rows = []
    for method, (x, intervened, ok, fb) in methods.items():
        pre_norm_x = np.asarray(x)
        x = preserve(pre_norm_x, d_raw, cfg["preserve_norm"])
        applied = -x
        residual = float(true_g @ x) < -1e-10
        first_order = float(true_g @ applied)
        delta_q = (.5 * float(applied @ applied) - float(mean_target @ applied)
                   + cfg["rho"] / 4 * float(np.sum((applied+.5)**4 - .5**4)))
        update_norm = float(np.linalg.norm(applied))
        row = {
            "case_id": cfg["case_id"], "scan": cfg["scan"], "x_name": cfg["x_name"],
            "x_value": cfg["x_value"], "population_seed": pop_seed,
            "probe_seed": probe_seed, "method": method,
            "angle": cfg["angle"], "gamma": cfg["gamma"], "eta": cfg["eta"],
            "epsilon": cfg["epsilon"], "rho": cfg["rho"], "sigma": cfg["sigma"],
            "N": N, "b": b, "m": m, "delta": cfg["delta"],
            "probe_fraction": m / (N - b),
            "prediction_query_count": 2 * r * (b + m),
            "true_conflict": int(conflict),
            "true_dot": true_dot,
            "subspace_dot": float(true_c @ a),
            "alignment_error": abs(true_dot - float(true_c @ a)),
            "coverage_fraction": coverage_fraction,
            "estimated_coverage_fraction": float(np.linalg.norm(c)/np.linalg.norm(true_g)),
            "coordinate_error": coord_error, "fd_error": fd_error,
            "estimation_error": est_error, "set_coverage": covered,
            "estimation_error_e1": float(c[0]-true_fd[0]),
            "estimation_error_e2": float(c[1]-true_fd[1]),
            "estimation_error_e3": float(c[2]-true_fd[2]),
            "released": int(not intervened), "intervened": int(intervened),
            "false_release": int(conflict and not intervened),
            "false_intervention": int((not conflict) and intervened),
            "residual_conflict": int(residual),
            "delta_q": delta_q, "q_final": q0 + delta_q,
            "delta_i": .5 * float(np.sum((applied-e1)**2)) - i0,
            "first_order_change": first_order,
            "positive_first_order_change": max(first_order, 0.0),
            "effective_correction": int(intervened and update_norm > 1e-10
                                        and not residual),
            "correction_distance": float(np.linalg.norm(x-d_raw)),
            "update_norm": update_norm,
            "zero_update": int(update_norm <= 1e-10),
            "solver_success": int(ok), "solver_fallback": int(fb),
            "pre_norm_descent_sparse": sparse_vector(pre_norm_x),
            "applied_update_sparse": sparse_vector(applied),
        }
        rows.append(row)
    return rows


def clustered_interval(rows, metric):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["population_seed"]].append(float(row[metric]))
    values = np.asarray([np.mean(v) for v in by_seed.values()])
    if len(values) < 2:
        return 0.0
    return float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) /
                 math.sqrt(len(values)))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["method"])].append(row)
    output = []
    for (case_id, method), group in grouped.items():
        ref = group[0]
        item = {k: ref[k] for k in (
            "case_id", "scan", "x_name", "x_value", "method", "angle", "gamma",
            "eta", "epsilon", "rho", "sigma", "N", "b", "m", "delta",
            "probe_fraction"
        )}
        item["n"] = len(group)
        item["populations"] = len({row["population_seed"] for row in group})
        item["true_conflict_count"] = sum(row["true_conflict"] for row in group)
        item["true_nonconflict_count"] = item["n"] - item["true_conflict_count"]
        for metric in METRICS:
            item[metric+"_mean"] = float(np.mean([row[metric] for row in group]))
            item[metric+"_ci95"] = clustered_interval(group, metric)
        for metric in ("delta_q", "estimated_coverage_fraction", "coordinate_error"):
            samples = np.array([row[metric] for row in group], dtype=float)
            item[metric+"_q05"] = float(np.quantile(samples, .05))
            item[metric+"_q95"] = float(np.quantile(samples, .95))
        item["false_release_count"] = sum(row["false_release"] for row in group)
        item["false_intervention_count"] = sum(row["false_intervention"] for row in group)
        item["false_release_rate"] = (
            item["false_release_count"]/item["true_conflict_count"]
            if item["true_conflict_count"] else math.nan
        )
        item["false_intervention_rate"] = (
            item["false_intervention_count"]/item["true_nonconflict_count"]
            if item["true_nonconflict_count"] else math.nan
        )
        item["false_release_rate_ci95"] = (
            item["false_release_ci95"] if item["true_conflict_count"] else math.nan
        )
        item["false_intervention_rate_ci95"] = (
            item["false_intervention_ci95"] if item["true_nonconflict_count"] else math.nan
        )
        output.append(item)
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(summary, path):
    fig, axs = plt.subplots(2, 3, figsize=(21.5, 12.5), constrained_layout=True)
    fig.get_layout_engine().set(wspace=.06, hspace=.06, w_pad=.08, h_pad=.08)
    def select(scan, method, **filters):
        return sorted(
            [x for x in summary if x["scan"] == scan and x["method"] == method
             and all(x[k] == v for k, v in filters.items())],
            key=lambda x: x["x_value"],
        )

    styles = {
        "Raw": ("#64748B", "-", "o"),
        "PILOT": ("#059669", "-", "*"),
    }
    ax = axs[0, 0]
    ax.axvspan(30, 90, color="#DCFCE7", alpha=.72, zorder=0)
    ax.axvspan(90, 150, color="#FEE2E2", alpha=.78, zorder=0)
    for method in ("Raw", "PILOT"):
        values = select("A_angle", method)
        color, line, marker = styles[method]
        xx = [x["angle"] for x in values]
        yy = np.array([x["delta_q_mean"] for x in values])
        low = [x["delta_q_q05"] for x in values]
        high = [x["delta_q_q95"] for x in values]
        ax.fill_between(xx, low, high, color=color, alpha=.16)
        ax.plot(xx, yy, linestyle=line, marker=marker, color=color,
                linewidth=8.5 if method == "PILOT" else 6.6,
                markersize=19 if method == "PILOT" else 14,
                label=method)
    ax.axhline(0, color="#374151", linewidth=3.4)
    ax.axvline(90, color="#6B7280", linestyle=":", linewidth=3.6)
    ax.set(title=r"(a) Conflict begins at $90^\circ$", xlabel="Gradient angle (degrees)",
           ylabel="Prediction loss change", xlim=(30, 150))
    ax.legend(loc="upper left", frameon=False, fontsize=29)

    ax = axs[1, 0]
    raw = select("D_locality", "Raw")
    pilot_full = select("D_full_probe", "PILOT")
    xx = np.array([x["eta"] for x in raw])
    raw_change = np.array([x["delta_q_mean"] for x in raw])
    full_change = np.array([x["delta_q_mean"] for x in pilot_full])
    ax.plot(xx, raw_change, "-o", color="#64748B", linewidth=8,
            markersize=16, label="Raw")
    ax.plot(xx, full_change, "-*", color="#059669", linewidth=8.5,
            markersize=19, label="PILOT")
    raw_slope = np.polyfit(np.log(xx), np.log(raw_change), 1)[0]
    full_slope = np.polyfit(np.log(xx), np.log(full_change), 1)[0]
    ax.text(.025, .004, f"slope ≈ {raw_slope:.2f}", color="#475569",
            fontsize=36, fontweight="medium")
    ax.text(.025, .00008, f"slope ≈ {full_slope:.2f}",
            color="#059669", fontsize=36, fontweight="medium")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="(d) PILOT removes first-order harm", xlabel="Update step size",
           ylabel="Prediction loss change", xlim=(.009, .105),
           ylim=(2e-5, .12))
    ax.set_xticks((.01, .02, .05, .1))
    ax.set_xticklabels(("0.01", "0.02", "0.05", "0.1"))
    ax.legend(loc="upper left", frameon=True, framealpha=.92,
              fontsize=29, handlelength=1.2, borderpad=.3)

    ax = axs[0, 1]
    coverage_scan = ("B_coverage_log" if any(
        x["scan"] == "B_coverage_log" for x in summary) else "B_coverage")
    values = select(coverage_scan, "PILOT")
    xx = [x["gamma"] for x in values]
    effective = np.array([x["effective_correction_mean"] for x in values])
    effective_ci = np.array([x["effective_correction_ci95"] for x in values])
    ax.fill_between(xx, np.maximum(0, effective-effective_ci),
                    np.minimum(1, effective+effective_ci),
                    color="#1D4ED8", alpha=.16)
    ax.plot(xx, effective, "-o", color="#1D4ED8", linewidth=8,
            markersize=16, label="Effective correction")
    ax.set(title="(b) Subspace coverage effect",
           xlabel="Lateral coverage γ", ylabel="Effective correction rate",
           ylim=(-.04, 1.05))
    if coverage_scan == "B_coverage_log":
        ax.set_xscale("log")
        ax.set_xlim(8e-5, 1.2)
    ax.tick_params(axis="y", colors="#1D4ED8")
    ax.yaxis.label.set_color("#1D4ED8")
    right_b = ax.twinx()
    pilot_loss = np.array([x["q_final_mean"] for x in values])
    right_b.plot(xx, pilot_loss, "-*", color="#059669", linewidth=8.5,
                 markersize=19, label="PILOT final loss")
    q_min = pilot_loss.min()
    q_max = pilot_loss.max()
    q_pad = .12 * (q_max - q_min)
    right_b.set(ylabel="Final forecast loss",
                ylim=(q_min-q_pad, q_max+q_pad))
    right_b.yaxis.label.set_color("#059669")

    ax = axs[0, 2]
    values = select("C_difference", "Raw")
    xx = np.array([x["epsilon"] for x in values])
    truncation = np.array([x["fd_error_mean"] for x in values])
    estimate = np.array([x["coordinate_error_mean"] for x in values])
    low = np.array([x["coordinate_error_q05"] for x in values])
    high = np.array([x["coordinate_error_q95"] for x in values])
    ax.loglog(xx, truncation, "-X", color="#A16207",
              markersize=16, linewidth=8, label="Difference error")
    ax.fill_between(xx, np.maximum(low, 1e-15), high,
                    color="#1D4ED8", alpha=.18)
    ax.loglog(xx, estimate, "-o", color="#1D4ED8",
              markersize=16, linewidth=8, label="Sampling error")
    fd_slope = np.polyfit(np.log(xx[(xx >= 1e-4) & (xx <= 3e-2)]),
                          np.log(truncation[(xx >= 1e-4) & (xx <= 3e-2)]), 1)[0]
    sample_slope = np.polyfit(np.log(xx), np.log(estimate), 1)[0]
    ax.text(2e-4, .001, f"slope ≈ {sample_slope:.2f}",
            color="#1D4ED8", fontsize=36)
    ax.text(1.5e-5, 1e-6, f"slope ≈ {fd_slope:.2f}",
            color="#92400E", fontsize=36)
    ax.legend(loc="lower right", frameon=True,
              framealpha=.9, fontsize=29)
    ax.set(title=r"(c) Difference error follows $O(\varepsilon^2)$",
           xlabel="Perturbation step size", ylabel="Coordinate error")

    ax = axs[1, 1]
    delta_colors = {.05: "#1D4ED8", .1: "#059669", .2: "#7C3AED"}
    for delta, color in delta_colors.items():
        values = select("E_probe", "PILOT", angle=120., sigma=.5, delta=delta)
        xx = [x["probe_fraction"] for x in values]
        yy = np.array([x["set_coverage_mean"] for x in values])
        ci = np.array([x["set_coverage_ci95"] for x in values])
        ax.fill_between(xx, yy-ci, yy+ci, color=color, alpha=.12)
        ax.plot(
            xx, yy, "-o", color=color, markersize=16, linewidth=8,
            label=f"δ={delta:g}",
        )
        ax.axhline(1 - delta, color=color, linestyle="--", linewidth=5.6,
                   alpha=.85, label="_nolegend_")
    ax.set(title="(e) Set coverage",
           xlabel="Probe rate",
           ylabel="Empirical set coverage", xlim=(0, 1.02), ylim=(.70, 1.03))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.legend(loc="lower center", ncol=3, frameon=True,
              framealpha=.95, fontsize=29, columnspacing=.3,
              handlelength=1.2, borderpad=.3)

    ax = axs[1, 2]
    angle_styles = {
        85.: ("#14532D", "-", "D"),
        89.: ("#16A34A", "--", "s"),
        90.: ("#4ADE80", ":", "o"),
        91.: ("#F87171", "--", "^"),
        95.: ("#991B1B", "-", "X"),
    }
    plotted = {}
    for angle in (85., 89., 90., 95., 91.):
        color, line, marker = angle_styles[angle]
        values = select("F_delta", "PILOT", angle=angle)
        plotted[angle] = ax.errorbar(
            [x["delta"] for x in values],
            [x["intervened_mean"] for x in values],
            yerr=[x["intervened_ci95"] for x in values],
            color=color, linestyle=line, marker=marker,
            markersize=16, linewidth=8, capsize=6,
            capthick=4, elinewidth=4, label=f"{angle:g}°",
        )
    # Matplotlib fills columns first; this order displays 85, 89, 90 above 91, 95.
    legend_angles = (85., 91., 89., 95., 90.)
    ax.legend([plotted[angle] for angle in legend_angles],
              [f"{angle:g}°" for angle in legend_angles],
              loc="center", bbox_to_anchor=(.5, .44), ncol=3,
              frameon=True, framealpha=.95, fontsize=29,
              columnspacing=.5, handlelength=1.2)
    ax.set(title="(f) Intervention near the boundary", xlabel="Confidence parameter δ",
           ylabel="Intervention rate", ylim=(-.03, 1.05))

    for axis in axs.flat:
        axis.grid(alpha=.22, linewidth=1.5)
        axis.tick_params(labelsize=32, width=2)
        axis.title.set_fontsize(36)
        axis.xaxis.label.set_fontsize(34)
        axis.yaxis.label.set_fontsize(34)
    right_b.tick_params(axis="y", labelsize=32, width=2,
                        colors="#059669", labelrotation=90, pad=2)
    right_b.yaxis.label.set_fontsize(34)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def probe_rate_diagnostic(summary, output_dir):
    angles = (85., 89., 90., 91., 95.)
    styles = {
        85.: ("#14532D", "-", "D"),
        89.: ("#16A34A", "--", "s"),
        90.: ("#4ADE80", ":", "o"),
        91.: ("#F87171", "--", "^"),
        95.: ("#991B1B", "-", "X"),
    }
    data = []
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for angle in angles:
        values = sorted(
            (row for row in summary if row["scan"] == "F_probe_rate"
             and row["method"] == "PILOT" and row["angle"] == angle),
            key=lambda row: row["probe_fraction"],
        )
        color, line, marker = styles[angle]
        ax.errorbar(
            [row["probe_fraction"] for row in values],
            [row["intervened_mean"] for row in values],
            yerr=[row["intervened_ci95"] for row in values],
            color=color, linestyle=line, marker=marker, linewidth=4,
            markersize=10, capsize=4, label=f"{angle:g}°",
        )
        data.extend({
            "angle": angle, "m": int(row["m"]),
            "probe_rate": row["probe_fraction"],
            "intervention_rate": row["intervened_mean"],
            "ci95": row["intervened_ci95"],
        } for row in values)
    ax.axvline(DEFAULTS["m"]/(DEFAULTS["N"]-DEFAULTS["b"]),
               color="#94A3B8", linestyle="--", linewidth=2.5)
    ax.set(title="Intervention rate vs probe rate (δ=0.1)",
           xlabel="Probe rate", ylabel="Intervention rate",
           xlim=(0, 1.03), ylim=(-.03, 1.05))
    ax.grid(alpha=.2)
    ax.tick_params(labelsize=18)
    ax.title.set_fontsize(22)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=17,
              frameon=False)
    fig.savefig(output_dir/"probe_rate_diagnostic.pdf", dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    write_csv(output_dir/"probe_rate_intervention.csv", data)

def report(summary, records):
    def metric(scan, method, key):
        return [x[key] for x in summary if x["scan"] == scan and x["method"] == method]
    c = sorted([x for x in summary if x["scan"] == "C_difference" and x["method"] == "Raw"],
               key=lambda x: x["epsilon"])
    eps = np.array([x["epsilon"] for x in c])
    err = np.array([x["fd_error_mean"] for x in c])
    usable = (eps >= 1e-4) & (eps <= 3e-2) & (err > 1e-15)
    slope = float(np.polyfit(np.log(eps[usable]), np.log(err[usable]), 1)[0])
    by_case = defaultdict(list)
    for row in records:
        if row["method"] == "Raw" and row["scan"] in ("E_probe", "F_delta"):
            by_case[row["case_id"]].append(row)
    max_bias = max(
        max(abs(float(np.mean([x[f"estimation_error_e{j}"] for x in group])))
            for j in (1, 2, 3))
        for group in by_case.values()
    )
    def focus(scan, angle, method, **filters):
        found = [x for x in summary if x["scan"] == scan and
                 x["angle"] == angle and x["method"] == method and
                 all(x[k] == v for k, v in filters.items())]
        return found[0]
    focus_filters = {"m": 256, "sigma": .5, "delta": .1}
    point = focus("E_probe", 120., "Point ZO", **focus_filters)
    robust = focus("E_probe", 120., "PILOT", **focus_filters)
    gate = focus("F_delta", 89., "PILOT", delta=.1)
    return {
        "maximum_absolute_coordinate_estimation_bias_by_case": max_bias,
        "focus_phi120_m256_sigma05": {
            "point_projection_residual_conflict_rate": point["residual_conflict_mean"],
            "robust_projection_residual_conflict_rate": robust["residual_conflict_mean"],
            "robust_projection_prediction_loss_change": robust["delta_q_mean"],
            "empirical_set_coverage": robust["set_coverage_mean"],
        },
        "focus_phi89_delta01": {
            "PILOT_intervention_rate": gate["intervened_mean"],
            "false_intervention_denominator": gate["true_nonconflict_count"],
        },
        "maximum_alignment_error": max(metric("A_angle", "Raw", "alignment_error_mean")),
        "finite_difference_log_slope": slope,
        "C_coordinate_estimation_error_range": [
            min(metric("C_difference", "Raw", "coordinate_error_mean")),
            max(metric("C_difference", "Raw", "coordinate_error_mean")),
        ],
        "E_empirical_set_coverage_range": [
            min(metric("E_probe", "PILOT", "set_coverage_mean")),
            max(metric("E_probe", "PILOT", "set_coverage_mean")),
        ],
        "F_empirical_set_coverage_range": [
            min(metric("F_delta", "PILOT", "set_coverage_mean")),
            max(metric("F_delta", "PILOT", "set_coverage_mean")),
        ],
        "PILOT_solver_fallback_count": int(sum(
            x["solver_fallback_mean"] * x["n"] for x in summary if x["method"] == "PILOT"
        )),
        "PILOT_residual_conflict_range": [
            min(metric("E_probe", "PILOT", "residual_conflict_mean")),
            max(metric("E_probe", "PILOT", "residual_conflict_mean")),
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/controlled_pilot_diagnostics_v5"))
    parser.add_argument("--population-seeds", type=int, default=DEFAULTS["population_seeds"])
    parser.add_argument("--probe-draws", type=int, default=DEFAULTS["probe_draws"])
    parser.add_argument("--refine-coverage", action="store_true",
                        help="run the log-spaced B scan and redraw the main figure")
    args = parser.parse_args()
    if args.population_seeds < 1 or args.probe_draws < 1:
        parser.error("Population seeds and probe draws must be positive")
    DEFAULTS["population_seeds"] = args.population_seeds
    DEFAULTS["probe_draws"] = args.probe_draws
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refine_coverage:
        base_summary_path = args.output_dir / "summary.csv"
        if not base_summary_path.exists():
            parser.error("--refine-coverage requires an existing summary.csv")
        plan = coverage_refinement_cases()
        with (args.output_dir / "coverage_log_config.json").open("w", encoding="utf-8") as h:
            json.dump({"defaults": DEFAULTS, "cases": plan, "methods": METHODS},
                      h, indent=2)
        rows = []
        for idx, cfg in enumerate(plan, 1):
            e1, V, q, d_raw = geometry(cfg)
            for pop_seed in range(cfg["population_count"]):
                targets, B, complement = population(cfg, pop_seed, V, q)
                q0 = loss_q(np.zeros(cfg["d"]), targets, cfg["rho"])
                for probe_seed in range(cfg["probe_draws"]):
                    rows.extend(evaluate_case(cfg, pop_seed, probe_seed, targets,
                                              B, complement, e1, V, q, d_raw, q0))
            print(f"Completed refined coverage {idx}/{len(plan)}", flush=True)
        write_csv(args.output_dir / "coverage_log_seed_records.csv", rows)
        refined = summarize(rows)
        by_case = {cfg["case_id"]: cfg for cfg in plan}
        for item in refined:
            item["d"] = by_case[item["case_id"]]["d"]
            item["rank"] = by_case[item["case_id"]]["rank"]
        write_csv(args.output_dir / "coverage_log_summary.csv", refined)
        figure(read_summary(base_summary_path) + refined,
               args.output_dir / "controlled_pilot_six_panel.pdf")
        return
    plan = cases()
    with (args.output_dir/"config.json").open("w", encoding="utf-8") as h:
        json.dump({
            "defaults": DEFAULTS, "cases": plan, "methods": METHODS,
            "initial_parameter": "zero vector",
            "imputation_target": "first standard basis vector",
            "subspace_basis": "first rank standard basis vectors",
            "population_rng_seed": "710000 + population_seed",
            "probe_rng_seed": "920000 + 1009 * population_seed + probe_seed",
            "population_seeds": list(range(args.population_seeds)),
            "probe_seeds": list(range(args.probe_draws)),
            "query_count_per_decision": "2 * rank * (batch_size + probe_size)",
        }, h, indent=2)
    rows = []
    completed = {}
    setting_names = ("d", "rank", "angle", "gamma", "eta", "epsilon", "rho",
                     "sigma", "N", "b", "m", "delta", "preserve_norm",
                     "population_count", "probe_draws")
    for idx,cfg in enumerate(plan, 1):
        key = tuple(cfg[name] for name in setting_names)
        if key in completed:
            for prior in completed[key]:
                row = prior.copy()
                row.update({name: cfg[name] for name in
                            ("case_id", "scan", "x_name", "x_value")})
                rows.append(row)
        else:
            first = len(rows)
            e1, V, q, d_raw = geometry(cfg)
            for pop_seed in range(cfg["population_count"]):
                targets, B, complement = population(cfg, pop_seed, V, q)
                q0 = loss_q(np.zeros(cfg["d"]), targets, cfg["rho"])
                for probe_seed in range(cfg["probe_draws"]):
                    rows.extend(evaluate_case(cfg, pop_seed, probe_seed, targets, B,
                                              complement, e1, V, q, d_raw, q0))
            completed[key] = rows[first:]
        if idx % 10 == 0 or idx == len(plan):
            print(f"Completed {idx}/{len(plan)} scan cells", flush=True)
    write_csv(args.output_dir/"seed_records.csv", rows)
    summary = summarize(rows)
    by_case = {cfg["case_id"]: cfg for cfg in plan}
    for item in summary:
        item["d"] = by_case[item["case_id"]]["d"]
        item["rank"] = by_case[item["case_id"]]["rank"]
    write_csv(args.output_dir/"summary.csv", summary)
    result = report(summary, rows)
    result["records"] = len(rows)
    result["conditions"] = len(plan)
    result["unique_experimental_settings"] = len(completed)
    with (args.output_dir/"report.json").open("w", encoding="utf-8") as h:
        json.dump(result, h, indent=2)
    refined_path = args.output_dir / "coverage_log_summary.csv"
    figure(summary + (read_summary(refined_path) if refined_path.exists() else []),
           args.output_dir/"controlled_pilot_six_panel.pdf")
    probe_rate_diagnostic(summary, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
