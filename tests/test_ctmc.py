"""Sanity checks for the closed-form station model."""

import numpy as np
import pytest

from src.ctmc import failure_rate, stationary_distribution


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
