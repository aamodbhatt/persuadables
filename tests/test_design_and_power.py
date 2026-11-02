"""SRM, balance, Holm, power and Cochran's Q, cross-checked against scipy and statsmodels."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower

from persuadables.stats.design import (
    balance_table,
    sample_ratio_mismatch,
    standardized_mean_difference,
)
from persuadables.stats.heterogeneity import cochran_q
from persuadables.stats.multiplicity import holm_adjust
from persuadables.stats.power import (
    minimum_detectable_effect,
    power_two_sample,
    sample_size_per_arm,
)


def test_srm_matches_scipy_chisquare() -> None:
    counts = (21306, 21307, 21387)
    ours = sample_ratio_mismatch(counts, (1 / 3, 1 / 3, 1 / 3))
    ref = stats.chisquare(counts, [sum(counts) / 3] * 3)
    assert ours.chi2 == pytest.approx(ref.statistic, rel=1e-12)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)


def test_srm_flags_a_broken_split() -> None:
    assert sample_ratio_mismatch((10_500, 9_500), (0.5, 0.5)).p_value < 1e-10


def test_srm_rejects_shares_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        sample_ratio_mismatch((10, 10), (0.5, 0.4))


def test_smd_hand_example() -> None:
    # means 2 and 1, sample variances 1 and 1 -> SMD = 1 / sqrt(1) = 1
    assert standardized_mean_difference(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])) == 1.0


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=40),
    st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=40),
    st.floats(0.1, 100),
    st.floats(-100, 100),
)
def test_smd_is_antisymmetric_and_affine_invariant(
    a: list[float], b: list[float], scale: float, shift: float
) -> None:
    x, y = np.array(a), np.array(b)
    if np.var(x, ddof=1) + np.var(y, ddof=1) < 1e-6:
        return
    smd = standardized_mean_difference(x, y)
    assert standardized_mean_difference(y, x) == pytest.approx(-smd, abs=1e-9)
    transformed = standardized_mean_difference(scale * x + shift, scale * y + shift)
    assert transformed == pytest.approx(smd, rel=1e-6, abs=1e-9)


def test_balance_table_covers_every_arm_and_covariate() -> None:
    rng = np.random.default_rng(0)
    arm = rng.integers(0, 3, 900)
    X = rng.normal(size=(900, 2))
    rows = balance_table(arm, X, ("a", "b"), control=0, treatments=(1, 2))
    assert [(r.arm, r.covariate) for r in rows] == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]


def test_holm_matches_statsmodels() -> None:
    rng = np.random.default_rng(1)
    p = np.r_[rng.uniform(0, 0.05, 6), rng.uniform(0, 1, 30), [0.0, 1.0, 0.03, 0.03]]
    _, expected, _, _ = multipletests(p, method="holm")
    np.testing.assert_allclose(holm_adjust(p), expected, rtol=0, atol=1e-15)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=50))
def test_holm_invariants(p_values: list[float]) -> None:
    p = np.array(p_values)
    adjusted = holm_adjust(p)
    assert np.all(adjusted >= p - 1e-15)
    assert np.all(adjusted <= 1.0)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-15)


@pytest.mark.parametrize(("n1", "n0"), [(21307, 21306), (5000, 15000), (300, 200)])
@pytest.mark.parametrize("sigma", [0.307, 0.0755, 15.8])
def test_mde_and_power_match_statsmodels(n1: int, n0: int, sigma: float) -> None:
    engine = NormalIndPower()
    mde = minimum_detectable_effect(sigma, n1, n0, alpha=0.05, power=0.8)
    effect_size = engine.solve_power(nobs1=n1, alpha=0.05, power=0.8, ratio=n0 / n1)
    # solve_power finds the effect size by numerical root-finding (tolerance ~1e-5).
    assert mde == pytest.approx(effect_size * sigma, rel=1e-4)
    power = power_two_sample(mde, sigma, n1, n0, alpha=0.05)
    assert power == pytest.approx(engine.power(mde / sigma, n1, 0.05, ratio=n0 / n1), rel=1e-10)
    assert power == pytest.approx(0.8, abs=1e-4)


def test_sample_size_round_trip() -> None:
    sigma, effect = math.sqrt(0.0057 * 0.9943), 0.00057  # a 10% lift on a 0.57% conversion rate
    n = sample_size_per_arm(effect, sigma, alpha=0.05, power=0.8)
    assert power_two_sample(effect, sigma, n, n, alpha=0.05) >= 0.8
    assert power_two_sample(effect, sigma, n - 50, n - 50, alpha=0.05) < 0.8


def test_cochran_q_matches_statsmodels() -> None:
    effects = np.array([0.071, 0.052, 0.098, 0.031])
    se = np.array([0.011, 0.009, 0.015, 0.012])
    ours = cochran_q(effects, se)
    ref = combine_effects(effects, se**2)
    assert ours.q == pytest.approx(ref.q, rel=1e-10)
    assert ours.df == 3
    assert ours.p_value == pytest.approx(stats.chi2.sf(ref.q, 3), rel=1e-10)


def test_cochran_q_is_zero_for_identical_effects_and_skips_zero_se() -> None:
    result = cochran_q(np.array([0.1, 0.1, 0.4]), np.array([0.02, 0.05, 0.0]))
    assert result.q == pytest.approx(0.0, abs=1e-12)
    assert result.levels_used == 2
