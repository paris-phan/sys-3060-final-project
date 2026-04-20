"""Closed-form M/M/1/K station model.

Notation follows the project spec:
    rho = mu / lambda   (deposit rate / withdrawal rate)
    pi_k = rho^k (1 - rho) / (1 - rho^(c+1))            for rho != 1
    pi_k = 1 / (c + 1)                                  for rho == 1

Long-run failure rate (stockouts at k=0 plus dockblocks at k=c):
    F_n(c) = lambda * pi_0 + mu * pi_c
"""

from __future__ import annotations

import numpy as np


def stationary_distribution(rho: float, c: int) -> np.ndarray:
    """Stationary distribution of the finite birth-death chain on {0,...,c}.

    Uses log-sum-exp normalization so the computation is stable for
    very small or very large rho (rho^(c+1) may underflow/overflow when
    evaluated directly but the normalized ratio is always well-defined).
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"rho must be a positive finite number, got {rho!r}")
    if not isinstance(c, (int, np.integer)) or int(c) != c or c < 0:
        raise ValueError(f"c must be a non-negative integer, got {c!r}")
    c = int(c)

    if np.isclose(rho, 1.0):
        return np.full(c + 1, 1.0 / (c + 1))

    # Unnormalized log-weights: log(rho^k) = k * log(rho).
    log_w = np.arange(c + 1) * np.log(rho)
    log_w -= log_w.max()  # shift for numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def failure_rate(lam: float, mu: float, c: int) -> float:
    """Long-run expected rate of user-facing failures at a station.

    F_n = lambda * pi_0 + mu * pi_c
        = (arrivals that stock out) + (returns that dockblock)
    """
    if not (np.isfinite(lam) and np.isfinite(mu)) or lam <= 0 or mu <= 0:
        raise ValueError(f"lam and mu must be positive and finite, got {lam!r}, {mu!r}")
    rho = mu / lam
    pi = stationary_distribution(rho, c)
    return float(lam * pi[0] + mu * pi[-1])
