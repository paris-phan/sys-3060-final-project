"""Sanity checks for the closed-form station model."""

import numpy as np
import pytest

from src.ctmc import (
    failure_rate,
    failure_rate_rebalanced,
    generator_matrix,
    stationary_distribution,
    stationary_distribution_rebalanced,
)


# ---------------------------------------------------------------------------
# stationary_distribution
# ---------------------------------------------------------------------------


def test_stationary_sums_to_one_across_regimes():
    for rho in [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 50.0]:
        for c in [1, 5, 20, 100]:
            pi = stationary_distribution(rho, c)
            assert pi.shape == (c + 1,)
            assert np.isclose(pi.sum(), 1.0, atol=1e-12), (rho, c, pi.sum())
            assert np.all(pi >= 0)


def test_stationary_rho_equal_one_is_uniform():
    pi = stationary_distribution(1.0, 10)
    assert np.allclose(pi, 1.0 / 11)


def test_stationary_small_rho_concentrates_on_zero():
    # rho << 1 => almost all mass at k=0 (empty station).
    pi = stationary_distribution(1e-3, 20)
    assert pi[0] > 0.99
    assert pi[-1] < 1e-50 or pi[-1] == 0.0


def test_stationary_large_rho_concentrates_on_capacity():
    # rho >> 1 => almost all mass at k=c (full station).
    pi = stationary_distribution(1e3, 20)
    assert pi[-1] > 0.99
    assert pi[0] < 1e-50 or pi[0] == 0.0


def test_stationary_matches_hand_computation_small_case():
    # c=2, rho=2: weights 1, 2, 4; normalized 1/7, 2/7, 4/7.
    pi = stationary_distribution(2.0, 2)
    expected = np.array([1.0, 2.0, 4.0]) / 7.0
    assert np.allclose(pi, expected)


def test_stationary_matches_hand_computation_rho_half():
    # c=3, rho=0.5: weights 1, 1/2, 1/4, 1/8; sum = 15/8; pi_k = weight / (15/8).
    pi = stationary_distribution(0.5, 3)
    weights = np.array([1.0, 0.5, 0.25, 0.125])
    expected = weights / weights.sum()
    assert np.allclose(pi, expected)


def test_stationary_monotone_in_rho():
    # For rho > 1, pi is strictly increasing in k; for rho < 1, strictly decreasing.
    pi_hi = stationary_distribution(3.0, 6)
    assert np.all(np.diff(pi_hi) > 0)
    pi_lo = stationary_distribution(1.0 / 3.0, 6)
    assert np.all(np.diff(pi_lo) < 0)


def test_stationary_rejects_bad_inputs():
    with pytest.raises(ValueError):
        stationary_distribution(0.0, 5)
    with pytest.raises(ValueError):
        stationary_distribution(-1.0, 5)
    with pytest.raises(ValueError):
        stationary_distribution(np.inf, 5)
    with pytest.raises(ValueError):
        stationary_distribution(1.0, -1)
    with pytest.raises(ValueError):
        stationary_distribution(1.0, 2.5)


def test_stationary_handles_extreme_rho_without_overflow():
    # Direct evaluation of rho^(c+1) would overflow; log-sum-exp should not.
    pi = stationary_distribution(1e6, 500)
    assert np.isfinite(pi).all()
    assert np.isclose(pi.sum(), 1.0)
    pi = stationary_distribution(1e-6, 500)
    assert np.isfinite(pi).all()
    assert np.isclose(pi.sum(), 1.0)


# ---------------------------------------------------------------------------
# failure_rate
# ---------------------------------------------------------------------------


def test_failure_rate_symmetric_case():
    # lam == mu => rho = 1, pi uniform; F = (lam + mu) / (c + 1).
    lam = mu = 10.0
    c = 9
    F = failure_rate(lam, mu, c)
    assert np.isclose(F, (lam + mu) / (c + 1))


def test_failure_rate_matches_direct_formula():
    lam, mu, c = 4.0, 6.0, 5
    pi = stationary_distribution(mu / lam, c)
    expected = lam * pi[0] + mu * pi[-1]
    assert np.isclose(failure_rate(lam, mu, c), expected)


def test_failure_rate_decreases_with_capacity_in_stable_regime():
    # For a typical imbalance ratio, adding docks should reduce the
    # failure rate (non-increasing marginal returns is the property
    # the greedy optimizer relies on).
    lam, mu = 10.0, 12.0
    F_values = [failure_rate(lam, mu, c) for c in range(2, 30)]
    diffs = np.diff(F_values)
    assert np.all(diffs <= 1e-12)  # monotone non-increasing


def test_failure_rate_bounded_by_min_rate_as_capacity_grows():
    # As c -> infty with rho != 1, one of the blocking probabilities
    # vanishes; F tends to min(lam, mu) * (limiting mass at that boundary).
    # Concretely with rho > 1 the chain concentrates at k=c so pi_0 -> 0
    # and pi_c -> 1 - 1/rho, giving F -> mu * (1 - lam/mu) = mu - lam.
    lam, mu, c = 5.0, 10.0, 200
    F = failure_rate(lam, mu, c)
    assert np.isclose(F, mu - lam, atol=1e-6)


def test_failure_rate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        failure_rate(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        failure_rate(1.0, 0.0, 5)
    with pytest.raises(ValueError):
        failure_rate(-1.0, 1.0, 5)
    with pytest.raises(ValueError):
        failure_rate(np.nan, 1.0, 5)


# ---------------------------------------------------------------------------
# Rebalanced chain (notebook 02 extension)
# ---------------------------------------------------------------------------


def test_generator_rows_sum_to_zero():
    for theta in [0.0, 0.5, 3.0, 100.0]:
        Q = generator_matrix(lam=2.0, mu=3.0, c=10, theta=theta, target=5)
        assert np.allclose(Q.sum(axis=1), 0.0, atol=1e-12)
        # Off-diagonals nonnegative, diagonal nonpositive.
        off = Q - np.diag(np.diag(Q))
        assert np.all(off >= -1e-15)
        assert np.all(np.diag(Q) <= 1e-15)


def test_generator_rejects_bad_inputs():
    with pytest.raises(ValueError):
        generator_matrix(lam=-1.0, mu=1.0, c=5, theta=0.1, target=2)
    with pytest.raises(ValueError):
        generator_matrix(lam=1.0, mu=-1.0, c=5, theta=0.1, target=2)
    with pytest.raises(ValueError):
        generator_matrix(lam=1.0, mu=1.0, c=5, theta=-0.1, target=2)
    with pytest.raises(ValueError):
        generator_matrix(lam=1.0, mu=1.0, c=5, theta=0.1, target=-1)
    with pytest.raises(ValueError):
        generator_matrix(lam=1.0, mu=1.0, c=5, theta=0.1, target=6)
    with pytest.raises(ValueError):
        generator_matrix(lam=np.inf, mu=1.0, c=5, theta=0.1, target=2)


@pytest.mark.parametrize("lam,mu,c", [
    (5.0, 2.0, 12),   # rho < 1  (source-like)
    (3.0, 3.0, 8),    # rho = 1  (balanced)
    (2.0, 6.0, 15),   # rho > 1  (sink-like)
])
def test_rebalanced_matches_closed_form_when_theta_zero(lam, mu, c):
    # With theta=0, the rebalancing column is never touched and we should
    # recover the closed-form geometric distribution. Check against the
    # base-model API across all three rho regimes.
    target = c // 2  # arbitrary; irrelevant when theta=0
    pi_reb = stationary_distribution_rebalanced(lam, mu, c, theta=0.0, target=target)
    rho = mu / lam
    pi_cf  = stationary_distribution(rho, c)
    assert pi_reb.shape == pi_cf.shape
    assert np.allclose(pi_reb, pi_cf, atol=1e-10), (
        f"max |diff| = {np.abs(pi_reb - pi_cf).max():.2e} for "
        f"lam={lam}, mu={mu}, c={c}"
    )


@pytest.mark.parametrize("theta", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("lam,mu,c", [
    (4.0, 2.0, 20),
    (3.0, 3.0, 10),
    (1.0, 5.0, 30),
])
def test_rebalanced_pi_is_valid_distribution(theta, lam, mu, c):
    target = c // 2
    pi = stationary_distribution_rebalanced(lam, mu, c, theta, target)
    assert pi.shape == (c + 1,)
    assert np.all(pi >= 0.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-12)


def test_failure_rate_rebalanced_monotone_nonincreasing_in_theta():
    # Core property the greedy optimizer relies on: more rebalancing
    # cannot hurt. Check at three representative stations.
    station_cases = [
        (4.0, 2.0, 20),   # source (rho < 1)
        (3.0, 3.0, 10),   # balanced
        (1.0, 8.0, 25),   # sink (rho > 1)
    ]
    theta_grid = np.geomspace(0.01, 50.0, 40)
    for lam, mu, c in station_cases:
        target = c // 2
        F = np.array([failure_rate_rebalanced(lam, mu, c, t, target)
                      for t in theta_grid])
        # Strict non-increasing (allow floating-point wiggle of 1e-10).
        diffs = np.diff(F)
        assert np.all(diffs <= 1e-10), (
            f"F not non-increasing for (lam={lam}, mu={mu}, c={c}); "
            f"max positive diff = {diffs.max():.3e}"
        )


def test_rebalanced_pi_concentrates_on_target_as_theta_huge():
    # theta -> infinity should drown out birth/death transitions and
    # concentrate pi on state t.
    lam, mu, c = 3.0, 5.0, 12
    for target in [0, 3, c // 2, c]:
        pi = stationary_distribution_rebalanced(lam, mu, c, theta=10_000.0, target=target)
        assert pi[target] > 0.999, (
            f"theta->inf should concentrate on target={target}; "
            f"got pi[target]={pi[target]:.6f}"
        )


def test_failure_rate_rebalanced_matches_base_when_theta_zero():
    # End-to-end sanity: failure_rate_rebalanced(theta=0) == failure_rate().
    for lam, mu, c in [(5.0, 2.0, 12), (3.0, 3.0, 8), (2.0, 6.0, 15)]:
        F_base = failure_rate(lam, mu, c)
        F_reb  = failure_rate_rebalanced(lam, mu, c, theta=0.0, target=c // 2)
        assert np.isclose(F_base, F_reb, atol=1e-10, rtol=1e-12)


def test_failure_rate_rebalanced_rejects_bad_inputs():
    with pytest.raises(ValueError):
        failure_rate_rebalanced(lam=0.0, mu=1.0, c=5, theta=0.1, target=2)
    with pytest.raises(ValueError):
        failure_rate_rebalanced(lam=1.0, mu=0.0, c=5, theta=0.1, target=2)
