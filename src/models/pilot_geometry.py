"""Euclidean projection for PILOT's ellipsoidal directional constraint."""

from __future__ import annotations

import numpy as np


def minimum_change_projection(raw, mean, covariance, beta):
    """Project onto mean @ x >= beta * sqrt(x @ covariance @ x).

    With one negative eigenvalue, the squared constraint is an elliptic cone.
    Orient its longitudinal eigenvector toward mean, then solve for the unique
    projected longitudinal coordinate by monotone bisection. Interior, polar,
    apex-only, and tangent-ray cases have direct solutions.
    """
    a = np.asarray(raw, dtype=np.float64)
    c = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("Invalid confidence radius")
    if c @ a >= beta * np.sqrt(max(0.0, a @ cov @ a)):
        return a.copy()
    if beta == 0:
        return a - (c @ a) / (c @ c) * c

    dual_direction = np.linalg.solve(cov, c)
    dual_sq = c @ dual_direction
    tol = 64 * np.finfo(float).eps * max(dual_sq, beta**2)
    if dual_sq < beta**2 - tol:
        return np.zeros_like(a)
    if abs(dual_sq - beta**2) <= tol:
        axis = dual_direction / np.linalg.norm(dual_direction)
        return max(0.0, a @ axis) * axis

    values, vectors = np.linalg.eigh(beta**2 * cov - np.outer(c, c))
    if values[0] >= 0:
        raise ArithmeticError("Could not resolve the negative axis of the PILOT cone")
    # At census the ridge can be many orders smaller than c c^T. Computing
    # transverse eigenvalues directly on the rank-one difference can lose all
    # their significant digits; first project the two terms separately.
    transverse = vectors[:, 1:]
    local_c = transverse.T @ c
    restricted = beta**2 * (transverse.T @ cov @ transverse) - np.outer(
        local_c, local_c
    )
    positive, rotation = np.linalg.eigh(restricted)
    if np.any(positive <= 0):
        raise ArithmeticError("Could not resolve the transverse axes of the PILOT cone")
    values[1:] = positive
    vectors[:, 1:] = transverse @ rotation
    if vectors[:, 0] @ c < 0:
        vectors[:, 0] *= -1
    rotated = vectors.T @ a
    longitudinal, lateral = rotated[0], rotated[1:]
    weights = values[1:] / -values[0]
    if longitudinal <= -np.linalg.norm(lateral / np.sqrt(weights)):
        return np.zeros_like(a)

    # At the boundary x_i = a_i * t / [t + w_i (t - a_0)].
    # Substitution into sum(w_i * x_i**2) = t**2 gives a monotone equation.
    low, high = max(0.0, longitudinal), np.linalg.norm(a)
    for _ in range(80):
        t = (low + high) / 2
        denom = t + weights * (t - longitudinal)
        if np.linalg.norm(np.sqrt(weights) * lateral / denom) > 1:
            low = t
        else:
            high = t
    t = high
    lateral_solution = lateral * t / (t + weights * (t - longitudinal))
    return vectors @ np.r_[t, lateral_solution]
