"""Effect estimates, cross-checked against statsmodels and scipy."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from persuadables.stats.inference import bootstrap_mean_diff, two_proportion_ztest, welch_ttest


@pytest.mark.parametrize(
    ("x1", "n1", "x0", "n0"),
    [(3895, 21307, 2262, 21306), (267, 21307, 122, 21306), (5, 40, 1, 55), (0, 10, 3, 12)],
)
def test_ztest_matches_statsmodels(x1: int, n1: int, x0: int, n0: int) -> None:
    ours = two_proportion_ztest(x1, n1, x0, n0, alpha=0.05)
    z, p = proportions_ztest([x1, x0], [n1, n0])
    low, high = confint_proportions_2indep(x1, n1, x0, n0, method="wald", compare="diff")
    assert ours.statistic == pytest.approx(z, rel=1e-10)
    assert ours.p_value == pytest.approx(p, rel=1e-8)
    assert ours.ci_low == pytest.approx(low, rel=1e-8)
    assert ours.ci_high == pytest.approx(high, rel=1e-8)


def test_ztest_with_no_events_reports_no_difference() -> None:
    result = two_proportion_ztest(0, 100, 0, 120)
    assert (result.diff, result.statistic, result.p_value) == (0.0, 0.0, 1.0)


def test_ztest_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="successes"):
        two_proportion_ztest(11, 10, 1, 10)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 5000), st.integers(1, 5000), st.floats(0, 1), st.floats(0, 1))
def test_ztest_is_antisymmetric(n1: int, n0: int, r1: float, r0: float) -> None:
    x1, x0 = round(r1 * n1), round(r0 * n0)
    forward = two_proportion_ztest(x1, n1, x0, n0)
    backward = two_proportion_ztest(x0, n0, x1, n1)
    assert forward.diff == pytest.approx(-backward.diff, abs=1e-12)
    assert forward.p_value == pytest.approx(backward.p_value, abs=1e-12)
    assert 0.0 <= forward.p_value <= 1.0
    assert forward.ci_low <= forward.diff <= forward.ci_high


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_welch_matches_scipy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    treatment = np.where(rng.random(3000) < 0.02, rng.gamma(2.0, 60.0, 3000), 0.0)
    control = np.where(rng.random(2500) < 0.01, rng.gamma(2.0, 60.0, 2500), 0.0)
    ours = welch_ttest(treatment, control, alpha=0.05)
    ref = stats.ttest_ind(treatment, control, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert ours.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert ours.ci_low == pytest.approx(ci.low, rel=1e-8)
    assert ours.ci_high == pytest.approx(ci.high, rel=1e-8)


def test_bootstrap_agrees_with_scipy_percentile_bootstrap() -> None:
    rng = np.random.default_rng(3)
    treatment = np.where(rng.random(4000) < 0.03, rng.gamma(2.0, 50.0, 4000), 0.0)
    control = np.where(rng.random(4000) < 0.015, rng.gamma(2.0, 50.0, 4000), 0.0)
    ours = bootstrap_mean_diff(treatment, control, resamples=4000, seed=11)
    ref = stats.bootstrap(
        (treatment, control),
        lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        n_resamples=4000,
        method="percentile",
        rng=np.random.default_rng(12),
    )
    # Two independent Monte Carlo runs: agree to within a few bootstrap standard errors / 10.
    tolerance = 0.1 * ours.se * 3
    assert ours.low == pytest.approx(ref.confidence_interval.low, abs=tolerance)
    assert ours.high == pytest.approx(ref.confidence_interval.high, abs=tolerance)
    assert ours.se == pytest.approx(ref.standard_error, rel=0.1)


def test_bootstrap_is_reproducible_and_covers_the_estimate() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.normal(1.0, 1.0, 500), rng.normal(0.0, 1.0, 600)
    first = bootstrap_mean_diff(a, b, resamples=500, seed=1)
    second = bootstrap_mean_diff(a, b, resamples=500, seed=1)
    assert first == second
    assert first.low < float(a.mean() - b.mean()) < first.high
