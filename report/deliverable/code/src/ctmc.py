"""Closed-form M/M/1/K station model + rebalancing extension.

Base model (birth-death only, closed form):
    rho = mu / lambda   (deposit rate / withdrawal rate)
    pi_k = rho^k (1 - rho) / (1 - rho^(c+1))            for rho != 1
    pi_k = 1 / (c + 1)                                  for rho == 1
    F_n(c) = lambda * pi_0 + mu * pi_c

Rebalanced model adds a third transition type: with rate theta, the state
jumps from any k != t to a fixed target level t. This makes the chain
non-birth-death (multi-state jumps), so the stationary distribution has
no closed form -- solve pi Q = 0 as a linear system instead.

# Citation
Developed alongside Claude Code (Opus 4.6)

"""

from __future__ import annotations

import numpy as np
from scipy import linalg as _sla


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


# ---------------------------------------------------------------------------
# Rebalancing-aware extension: added in notebook 02 without touching the
# base-model API above. A rebalancing truck visits station n at rate theta
# and resets the bike count to the target level t (floor(c/2) by default).
# Model:
#   Q[k, k+1] = mu                (deposit)        k = 0..c-1
#   Q[k, k-1] = lambda            (withdrawal)     k = 1..c
#   Q[k, t] += theta              (truck reset)    all k != t
#   Q[k, k]  = -sum_{j != k} Q[k, j]
# ---------------------------------------------------------------------------


def generator_matrix(
    lam: float, mu: float, c: int, theta: float, target: int
) -> np.ndarray:
    """Build the (c+1) x (c+1) generator Q for the rebalanced station chain.

    Rows index the current state k in {0, ..., c}. Row sums are zero.
    """
    if not (np.isfinite(lam) and np.isfinite(mu) and np.isfinite(theta)):
        raise ValueError(f"rates must be finite, got lam={lam}, mu={mu}, theta={theta}")
    if lam < 0 or mu < 0 or theta < 0:
        raise ValueError(f"rates must be nonnegative, got lam={lam}, mu={mu}, theta={theta}")
    if not isinstance(c, (int, np.integer)) or int(c) != c or c < 0:
        raise ValueError(f"c must be a non-negative integer, got {c!r}")
    c = int(c)
    if not isinstance(target, (int, np.integer)) or int(target) != target:
        raise ValueError(f"target must be integer, got {target!r}")
    target = int(target)
    if target < 0 or target > c:
        raise ValueError(f"target {target} must lie in [0, {c}]")

    K = c + 1
    Q = np.zeros((K, K))
    # Deposits: k -> k+1 at rate mu, for k = 0..c-1.
    if c > 0 and mu > 0:
        idx = np.arange(c)
        Q[idx, idx + 1] = mu
    # Withdrawals: k -> k-1 at rate lambda, for k = 1..c.
    if c > 0 and lam > 0:
        idx = np.arange(1, K)
        Q[idx, idx - 1] = lam
    # Rebalancing reset: every k != target jumps to target at rate theta.
    if theta > 0:
        rows = np.arange(K) != target
        Q[rows, target] += theta
    # Diagonal makes every row sum to zero.
    Q[np.arange(K), np.arange(K)] = -Q.sum(axis=1)
    return Q


def stationary_distribution_rebalanced(
    lam: float, mu: float, c: int, theta: float, target: int
) -> np.ndarray:
    """Stationary distribution pi of the rebalanced chain via pi Q = 0.

    Standard trick: solve (Q^T) pi = 0 with the normalization sum(pi) = 1
    by replacing one row of Q^T with all ones and the matching RHS with 1.
    For theta = 0 this reproduces the base-model closed form to within
    floating-point (checked by tests).
    """
    Q = generator_matrix(lam, mu, c, theta, target)
    K = Q.shape[0]
    A = Q.T.copy()
    b = np.zeros(K)
    # Replace the first row with the normalization constraint.
    A[0, :] = 1.0
    b[0] = 1.0
    try:
        pi = _sla.solve(A, b, check_finite=True)
    except _sla.LinAlgError as exc:
        raise RuntimeError(
            f"Singular generator system for (lam={lam}, mu={mu}, c={c}, "
            f"theta={theta}, target={target}): {exc}"
        ) from exc
    # Tiny negative values are floating-point noise; anything larger is a bug.
    if pi.min() < -1e-9:
        raise RuntimeError(
            f"Stationary distribution has negative entries (min={pi.min():.3e}) "
            f"for (lam={lam}, mu={mu}, c={c}, theta={theta}, target={target})."
        )
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Stationary distribution sum = {s}; solver failed.")
    return pi / s


def failure_rate_rebalanced(
    lam: float, mu: float, c: int, theta: float, target: int
) -> float:
    """F_n(theta) = lambda * pi_0(theta) + mu * pi_c(theta)."""
    if lam <= 0 or mu <= 0:
        raise ValueError(f"lam and mu must be positive, got {lam!r}, {mu!r}")
    pi = stationary_distribution_rebalanced(lam, mu, c, theta, target)
    return float(lam * pi[0] + mu * pi[-1])
