"""Greedy marginal-analysis optimizer for the rebalancing-rate budget.

Problem
-------
Given a cluster of N stations with per-station parameters (lambda_n, mu_n, c_n,
target_n), we allocate a total rebalancing budget `theta_total` across them
to minimize the cluster-wide long-run failure rate:

    F_total(theta_1, ..., theta_N) = sum_n F_n(theta_n)
    subject to: sum_n theta_n = theta_total, theta_n >= 0.

Each F_n(theta_n) is convex-decreasing in theta_n (strictly non-increasing
marginal returns; verified numerically in tests), so the separable problem
is exactly solved by **greedy marginal analysis** on a step-size grid:
repeatedly allocate one step of size `grid_resolution` to whichever station
currently has the largest marginal reduction -F_n'(theta_n).

The API:
    * `failure_vs_theta_curve`  - response curve per station over a theta grid.
    * `optimize_theta_budget`   - greedy optimum at one theta_total.
    * `pareto_frontier`         - optima across a grid of theta_total values
                                  (reuses one greedy pass for efficiency; the
                                  nested prefix structure means sampling the
                                  trace is equivalent to re-running at each
                                  budget).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.ctmc import failure_rate_rebalanced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_targets(stations_df: pd.DataFrame) -> list[int]:
    """Target level t_n = floor(c_n / 2), per the project spec."""
    return [int(c) // 2 for c in stations_df["c"]]


def _station_F(lam: float, mu: float, c: int, theta: float, target: int) -> float:
    return failure_rate_rebalanced(float(lam), float(mu), int(c),
                                    float(theta), int(target))


# ---------------------------------------------------------------------------
# Per-station response curves
# ---------------------------------------------------------------------------


def failure_vs_theta_curve(
    stations_df: pd.DataFrame,
    theta_grid: Iterable[float],
    target_levels: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Long-format curve F_n(theta) for each station over `theta_grid`.

    Parameters
    ----------
    stations_df : must contain columns {station_id, lam, mu, c}. Other
                  columns (name, etc.) are passed through.
    theta_grid  : iterable of non-negative theta values.
    target_levels : optional per-station targets. Defaults to floor(c/2).

    Returns
    -------
    DataFrame with columns (station_id, name [if present], theta, F_n,
    pi_0, pi_c). One row per (station, theta).
    """
    # Local import so tests don't pay the cost unless they hit this path.
    from src.ctmc import stationary_distribution_rebalanced

    stations = stations_df.reset_index(drop=True)
    theta_grid = np.asarray(list(theta_grid), dtype=float)
    targets = list(target_levels) if target_levels is not None \
              else _default_targets(stations)

    rows = []
    has_name = "name" in stations.columns
    for i, s in stations.iterrows():
        lam, mu, c = float(s["lam"]), float(s["mu"]), int(s["c"])
        target = int(targets[i])
        for th in theta_grid:
            pi = stationary_distribution_rebalanced(lam, mu, c, float(th), target)
            F  = lam * pi[0] + mu * pi[-1]
            row = {
                "station_id": s["station_id"],
                "theta": float(th),
                "F_n":   float(F),
                "pi_0":  float(pi[0]),
                "pi_c":  float(pi[-1]),
                "target": target,
            }
            if has_name:
                row["name"] = s["name"]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-budget optimizer (greedy marginal analysis)
# ---------------------------------------------------------------------------


def optimize_theta_budget(
    stations_df: pd.DataFrame,
    theta_total: float,
    grid_resolution: float = 0.01,
    target_levels: Optional[Iterable[int]] = None,
    verify_monotone: bool = True,
) -> dict:
    """Greedy allocation of a rebalancing-rate budget across stations.

    Discretize each theta_n in steps of `grid_resolution`. Starting from
    theta_n = 0 for all n, repeatedly allocate one step to whichever
    station's F_n drops the most. Stop when the cumulative budget equals
    `theta_total`.

    Returns
    -------
    dict with keys:
        theta_optimal     : ndarray (N,)   - per-station optimal theta.
        F_by_station      : ndarray (N,)   - F_n at the optimum.
        F_total           : float          - sum of F_by_station.
        allocation_trace  : list of (step, station_id, theta_total_so_far,
                                      F_total_so_far).
        theta_history     : ndarray (steps+1, N) - theta vector after each step
                                                   (row 0 = initial zeros).
        F_total_history   : ndarray (steps+1,) - F_total after each step.
        n_violations      : int            - steps at which a station's
                                              marginal gain INCREASED (would
                                              invalidate the greedy step).
    """
    if theta_total < 0:
        raise ValueError(f"theta_total must be nonnegative, got {theta_total}")
    if grid_resolution <= 0:
        raise ValueError(f"grid_resolution must be positive, got {grid_resolution}")

    stations = stations_df.reset_index(drop=True)
    n = len(stations)
    if n == 0:
        raise ValueError("stations_df is empty")

    targets = list(target_levels) if target_levels is not None \
              else _default_targets(stations)
    if len(targets) != n:
        raise ValueError(f"target_levels length {len(targets)} != n stations {n}")

    lams = stations["lam"].to_numpy(dtype=float)
    mus  = stations["mu"].to_numpy(dtype=float)
    cs   = stations["c"].to_numpy(dtype=int)
    sids = stations["station_id"].tolist()

    theta = np.zeros(n, dtype=float)
    # Initial F at theta = 0 per station.
    F_n = np.array([_station_F(lams[i], mus[i], cs[i], 0.0, targets[i])
                    for i in range(n)])

    def _marginal(i: int) -> float:
        return F_n[i] - _station_F(lams[i], mus[i], cs[i],
                                    theta[i] + grid_resolution, targets[i])

    marginals = np.array([_marginal(i) for i in range(n)])

    total_steps = int(round(theta_total / grid_resolution))
    trace = []
    theta_history = np.zeros((total_steps + 1, n), dtype=float)
    F_total_history = np.zeros(total_steps + 1, dtype=float)
    F_total_history[0] = float(F_n.sum())
    prev_marg_at_station = np.full(n, np.inf)  # for monotone check
    n_violations = 0

    for step in range(total_steps):
        i_best = int(np.argmax(marginals))
        dF = float(marginals[i_best])
        if dF <= 0.0:
            # No further reduction is possible at this resolution. All
            # remaining budget would still be valid to allocate (monotone
            # non-increasing property means it won't make things worse),
            # but we stop spreading since it's pointless.
            break
        # Check non-increasing marginal-returns property on this station.
        if verify_monotone and dF > prev_marg_at_station[i_best] + 1e-10:
            n_violations += 1
        prev_marg_at_station[i_best] = dF

        theta[i_best] += grid_resolution
        F_n[i_best] -= dF
        marginals[i_best] = _marginal(i_best)

        theta_history[step + 1] = theta
        F_total_history[step + 1] = float(F_n.sum())
        trace.append((step + 1, sids[i_best],
                      float((step + 1) * grid_resolution),
                      float(F_n.sum())))

    # If we broke out early, truncate histories and pad theta_total accounting.
    actual_steps = len(trace)
    theta_history = theta_history[: actual_steps + 1]
    F_total_history = F_total_history[: actual_steps + 1]

    if verify_monotone and n_violations:
        print(f"WARNING: {n_violations} monotonicity violations detected "
              f"(marginal return increased for a station across successive "
              f"picks). Check the model or tighten grid_resolution.")

    return {
        "theta_optimal":    theta,
        "F_by_station":     F_n,
        "F_total":          float(F_n.sum()),
        "allocation_trace": trace,
        "theta_history":    theta_history,
        "F_total_history":  F_total_history,
        "n_violations":     int(n_violations),
    }


# ---------------------------------------------------------------------------
# Pareto frontier over a grid of budgets
# ---------------------------------------------------------------------------


def pareto_frontier(
    stations_df: pd.DataFrame,
    theta_total_grid: Iterable[float],
    grid_resolution: float = 0.01,
    target_levels: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Pareto frontier over a grid of total budgets.

    Exploits the nested prefix structure of greedy marginal analysis: if
    we run once at `max(theta_total_grid)`, the optimal allocation at any
    smaller budget is a prefix of that run. So we compute the greedy trace
    once and sample the per-budget optima from it. This is exactly
    equivalent to re-running `optimize_theta_budget` at each grid point
    when the non-increasing-returns property holds (verified by the
    greedy itself).

    Returns
    -------
    DataFrame with columns:
      theta_total        : the sampled budget (may differ from the grid
                           value by < grid_resolution).
      F_total            : optimum total failure rate at that budget.
      theta_<station_id> : per-station optimal theta (one column each).
      F_<station_id>     : per-station F_n at the optimum (one column each).
    """
    theta_total_grid = np.sort(np.asarray(list(theta_total_grid), dtype=float))
    if theta_total_grid.min() < 0:
        raise ValueError("theta_total_grid must be nonnegative")
    theta_max = float(theta_total_grid[-1])

    stations = stations_df.reset_index(drop=True)
    sids = stations["station_id"].tolist()
    targets = list(target_levels) if target_levels is not None \
              else _default_targets(stations)

    # Run the greedy once up to the max budget.
    result = optimize_theta_budget(
        stations, theta_max, grid_resolution, target_levels=targets
    )
    theta_history   = result["theta_history"]
    F_total_history = result["F_total_history"]

    # Per-station F at each recorded step. We need this to populate F_<sid>.
    n = len(stations)
    lams = stations["lam"].to_numpy(dtype=float)
    mus  = stations["mu"].to_numpy(dtype=float)
    cs   = stations["c"].to_numpy(dtype=int)

    def _F_at(theta_vec: np.ndarray) -> np.ndarray:
        return np.array([
            _station_F(lams[i], mus[i], cs[i], theta_vec[i], int(targets[i]))
            for i in range(n)
        ])

    rows = []
    for target_total in theta_total_grid:
        idx = min(int(round(target_total / grid_resolution)),
                  len(F_total_history) - 1)
        theta_vec = theta_history[idx]
        F_vec = _F_at(theta_vec)
        row = {
            "theta_total": float(idx * grid_resolution),
            "F_total":     float(F_total_history[idx]),
        }
        for j, sid in enumerate(sids):
            row[f"theta_{sid}"] = float(theta_vec[j])
            row[f"F_{sid}"]     = float(F_vec[j])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Knee detection (kneedle method, convex decreasing curve)
# ---------------------------------------------------------------------------


def find_knee(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the knee on a monotone-decreasing convex curve.

    Uses the kneedle approach: normalize both axes to [0, 1], then find the
    point with the largest vertical drop below the chord connecting the
    endpoints (endpoints are (0, 1) and (1, 0) after normalizing a
    decreasing curve). For a convex decreasing curve this is the classical
    'elbow.'
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("x and y must be 1D arrays of the same length >= 3")
    x_rng = max(x.max() - x.min(), 1e-12)
    y_rng = max(y.max() - y.min(), 1e-12)
    x_n = (x - x.min()) / x_rng
    y_n = (y - y.min()) / y_rng
    # Endpoints after normalization: (0, 1) and (1, 0) because y decreases
    # from y.max() to y.min(). Chord: y_line = 1 - x. Drop below chord:
    chord_drop = (1.0 - x_n) - y_n
    return int(np.argmax(chord_drop))
