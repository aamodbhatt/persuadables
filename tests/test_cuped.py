"""CUPED: equivalence with regression adjustment, variance reduction and unbiasedness."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from persuadables.stats.cuped import cuped_adjust, cuped_effect, cuped_theta


def _experiment(
    n: int, seed: int, effect: float, signal: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.gamma(2.0, 50.0, n), rng.integers(1, 13, n).astype(float)])
    treated = rng.random(n) < 0.5
    y = signal * (0.01 * X[:, 0] - 0.2 * X[:, 1]) + effect * treated + rng.normal(0, 1, n)
    return y, X, treated


def test_single_covariate_theta_is_cov_over_var() -> None:
    y, X, _ = _experiment(2000, 0, 0.3, 1.0)
    expected = np.cov(y, X[:, 0])[0, 1] / np.var(X[:, 0], ddof=1)
    assert cuped_theta(y, X[:, :1])[0] == pytest.approx(expected, rel=1e-10)


def test_cuped_matches_ancova_regression() -> None:
    """With randomization, CUPED and OLS of y on (treatment, centred covariates) agree closely."""
    y, X, treated = _experiment(20_000, 1, 0.25, 1.0)
    ours = cuped_effect(y, X, treated)
    design = sm.add_constant(np.column_stack([treated.astype(float), X - X.mean(axis=0)]))
    ols = sm.OLS(y, design).fit(cov_type="HC2")
    assert ours.adjusted.diff == pytest.approx(ols.params[1], abs=0.05 * ols.bse[1])
    assert ours.adjusted.se == pytest.approx(ols.bse[1], rel=0.02)


def test_cuped_shrinks_the_standard_error_when_covariates_predict() -> None:
    y, X, treated = _experiment(10_000, 2, 0.2, 3.0)
    result = cuped_effect(y, X, treated)
    assert result.variance_reduction > 0.5
    assert result.adjusted.se / result.raw.se < 0.75


def test_cuped_is_unbiased() -> None:
    estimates = [
        cuped_effect(*_experiment(1500, seed, 0.3, 2.0)).adjusted.diff for seed in range(200)
    ]
    # 200 draws of an estimator with SE ~0.04: the mean is within ~4 standard errors of the truth.
    assert np.mean(estimates) == pytest.approx(0.3, abs=4 * np.std(estimates) / np.sqrt(200))


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.0, 5.0))
def test_pooled_variance_never_increases(seed: int, signal: float) -> None:
    y, X, _ = _experiment(400, seed, 0.1, signal)
    adjusted = cuped_adjust(y, X, cuped_theta(y, X))
    assert np.var(adjusted) <= np.var(y) * (1 + 1e-12)
    assert np.mean(adjusted) == pytest.approx(np.mean(y), abs=1e-9)
