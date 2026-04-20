"""Tests for src/optimization.py.

The key test (#1) against the base-model CSV exercises the entire stack
end-to-end: optimize_theta_budget with theta_total=0 should reproduce
F_base exactly (modulo floating point).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ctmc import failure_rate, failure_rate_rebalanced
from src.optimization import (
    failure_vs_theta_curve,
    find_knee,
    optimize_theta_budget,
    pareto_frontier,
)

ROOT = Path(__file__).resolve().parent.parent
BASE_CSV = ROOT / "data" / "processed" / "base_model_results.csv"


def _toy_cluster() -> pd.DataFrame:
    # Small synthetic 3-station cluster: one source, one balanced, one sink.
    return pd.DataFrame({
        "station_id": ["S1", "S2", "S3"],
        "name":       ["source", "balanced", "sink"],
        "lam": [6.0, 3.0, 1.0],
        "mu":  [2.0, 3.0, 5.0],
        "c":   [20, 10, 15],
    })


# ---------------------------------------------------------------------------
# failure_vs_theta_curve
# ---------------------------------------------------------------------------


def test_curve_theta_zero_matches_base():
    stations = _toy_cluster()
    curve = failure_vs_theta_curve(stations, theta_grid=[0.0])
    # F at theta=0 should equal the closed-form base failure rate.
    for _, r in curve.iterrows():
        s = stations.loc[stations["station_id"] == r["station_id"]].iloc[0]
        expected = failure_rate(float(s["lam"]), float(s["mu"]), int(s["c"]))
        assert np.isclose(r["F_n"], expected, atol=1e-10, rtol=1e-12)


def test_curve_is_monotone_in_theta():
    stations = _toy_cluster()
    thetas = np.geomspace(1e-3, 100.0, 40)
    curve = failure_vs_theta_curve(stations, thetas)
    for sid, g in curve.groupby("station_id"):
        F = g.sort_values("theta")["F_n"].to_numpy()
        assert np.all(np.diff(F) <= 1e-10), f"F not monotone on {sid}"


# ---------------------------------------------------------------------------
# optimize_theta_budget
# ---------------------------------------------------------------------------


def test_budget_zero_reproduces_base_F_on_toy():
    stations = _toy_cluster()
    res = optimize_theta_budget(stations, theta_total=0.0, grid_resolution=0.01)
    assert np.allclose(res["theta_optimal"], 0.0)
    F_expected = sum(
        failure_rate(float(s["lam"]), float(s["mu"]), int(s["c"]))
        for _, s in stations.iterrows()
    )
    assert np.isclose(res["F_total"], F_expected, atol=1e-10, rtol=1e-12)


@pytest.mark.skipif(not BASE_CSV.exists(),
                    reason="run notebook 01 first to produce base_model_results.csv")
def test_budget_zero_reproduces_F_base_from_csv():
    # Acceptance test #1 in the spec: solver at theta_total=0 reproduces
    # F_base from data/processed/base_model_results.csv.
    stations = pd.read_csv(BASE_CSV)
    res = optimize_theta_budget(stations, theta_total=0.0, grid_resolution=0.01)
    assert np.allclose(res["theta_optimal"], 0.0)
    F_base_csv = float(stations["F_n"].sum())
    assert np.isclose(res["F_total"], F_base_csv, atol=1e-8, rtol=1e-10), (
        f"CSV F_base = {F_base_csv:.6f}; optimizer at 0 = {res['F_total']:.6f}"
    )


def test_budget_allocates_exactly():
    stations = _toy_cluster()
    for theta_total in [1.0, 5.0, 25.0]:
        res = optimize_theta_budget(stations, theta_total, grid_resolution=0.01)
        # Sum of allocated thetas should be at most theta_total and within
        # one step of it (could stop early if marginal gain hits 0).
        assert np.sum(res["theta_optimal"]) <= theta_total + 1e-9
        assert np.sum(res["theta_optimal"]) >= theta_total - 0.02 \
            or res["F_total"] < 1e-10


def test_budget_F_total_monotone_in_budget():
    # Acceptance test #2: F_total is non-increasing as the budget grows.
    stations = _toy_cluster()
    budgets = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    F = [optimize_theta_budget(stations, b, grid_resolution=0.05)["F_total"]
         for b in budgets]
    diffs = np.diff(F)
    assert np.all(diffs <= 1e-9), f"F_total not monotone: {F}"


def test_budget_huge_drives_F_to_zero():
    # Acceptance test #3: very large budget drives F_total toward ~0.
    # Trucks arrive so fast every state almost instantly resets to target,
    # and target is a non-boundary state, so pi_0 and pi_c both go to 0.
    stations = _toy_cluster()
    res = optimize_theta_budget(stations, theta_total=500.0,
                                  grid_resolution=0.1)
    # Should be much smaller than the base failure rate (which is ~6-7/h
    # for this toy cluster).
    F_base = sum(
        failure_rate(float(s["lam"]), float(s["mu"]), int(s["c"]))
        for _, s in stations.iterrows()
    )
    assert res["F_total"] < 0.05 * F_base, (
        f"F_total={res['F_total']:.4f} at budget 500 vs F_base={F_base:.4f}"
    )


def test_budget_reports_no_monotonicity_violations_on_toy():
    # For a well-behaved convex F_n, the greedy should encounter zero
    # violations of the non-increasing-returns property.
    stations = _toy_cluster()
    res = optimize_theta_budget(stations, theta_total=10.0, grid_resolution=0.01)
    assert res["n_violations"] == 0


def test_budget_trace_is_consistent():
    stations = _toy_cluster()
    res = optimize_theta_budget(stations, theta_total=3.0, grid_resolution=0.01)
    trace = res["allocation_trace"]
    # theta_total_so_far and F_total_so_far in the trace should match
    # the theta_history / F_total_history arrays.
    for (step, _sid, tot, F_tot) in trace:
        assert np.isclose(tot, step * 0.01)
        assert np.isclose(F_tot, res["F_total_history"][step])
    # Final entry matches returned F_total.
    if trace:
        assert np.isclose(trace[-1][3], res["F_total"])


def test_budget_rejects_bad_inputs():
    stations = _toy_cluster()
    with pytest.raises(ValueError):
        optimize_theta_budget(stations, theta_total=-1.0)
    with pytest.raises(ValueError):
        optimize_theta_budget(stations, theta_total=1.0, grid_resolution=0.0)
    with pytest.raises(ValueError):
        optimize_theta_budget(stations.iloc[:0], theta_total=1.0)


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------


def test_pareto_is_monotone_in_theta_total():
    stations = _toy_cluster()
    grid = np.linspace(0.0, 20.0, 11)
    pf = pareto_frontier(stations, grid, grid_resolution=0.05)
    F = pf.sort_values("theta_total")["F_total"].to_numpy()
    assert np.all(np.diff(F) <= 1e-9)


def test_pareto_zero_matches_base():
    stations = _toy_cluster()
    pf = pareto_frontier(stations, [0.0, 1.0, 5.0], grid_resolution=0.05)
    row0 = pf.loc[pf["theta_total"] == pf["theta_total"].min()].iloc[0]
    F_base = sum(
        failure_rate(float(s["lam"]), float(s["mu"]), int(s["c"]))
        for _, s in stations.iterrows()
    )
    assert np.isclose(row0["F_total"], F_base, atol=1e-10)
    for sid in stations["station_id"]:
        assert row0[f"theta_{sid}"] == 0.0


def test_pareto_sums_match_theta_total_within_resolution():
    stations = _toy_cluster()
    grid = [0.0, 1.0, 4.0, 10.0]
    pf = pareto_frontier(stations, grid, grid_resolution=0.05)
    theta_cols = [f"theta_{sid}" for sid in stations["station_id"]]
    for _, r in pf.iterrows():
        allocated = float(r[theta_cols].sum())
        # allocated == theta_total up to one step (0.05)
        assert allocated <= r["theta_total"] + 1e-9
        assert allocated >= r["theta_total"] - 0.06 \
            or r["F_total"] < 1e-10


# ---------------------------------------------------------------------------
# find_knee
# ---------------------------------------------------------------------------


def test_find_knee_detects_obvious_elbow():
    # Convex decreasing curve with a sharp corner at x=2.
    x = np.linspace(0, 10, 101)
    y = np.where(x <= 2, 10.0 - 4.0 * x, 2.0 - 0.05 * (x - 2))
    idx = find_knee(x, y)
    assert 1.5 <= x[idx] <= 2.5


def test_find_knee_on_exp_decay_is_in_expected_zone():
    # F = exp(-x) has a knee around x=1-2.
    x = np.linspace(0, 10, 201)
    y = np.exp(-x)
    idx = find_knee(x, y)
    assert 0.5 <= x[idx] <= 3.0
