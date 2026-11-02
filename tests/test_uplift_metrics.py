"""Qini/uplift curves against a brute-force reference, plus their invariants."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from persuadables.uplift.metrics import (
    auuc,
    calibration_by_bin,
    perfect_qini_curve,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_curve,
)


def _reference_curves(
    score: np.ndarray, treated: np.ndarray, y: np.ndarray
) -> tuple[list[float], list[float], list[float]]:
    """Literal definition: for every distinct threshold, target everyone scoring >= it."""
    xs, qs, us = [0.0], [0.0], [0.0]
    for threshold in sorted(set(score.tolist()), reverse=True):
        top = score >= threshold
        n_t, n_c = (top & treated).sum(), (top & ~treated).sum()
        y_t, y_c = y[top & treated].sum(), y[top & ~treated].sum()
        xs.append(float(top.sum()))
        qs.append(float(y_t - (y_c * n_t / n_c if n_c else 0.0)))
        rate_t = y_t / n_t if n_t else 0.0
        rate_c = y_c / n_c if n_c else 0.0
        us.append(float((rate_t - rate_c) * top.sum()))
    return xs, qs, us


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(5, 120), st.integers(2, 8))
def test_curves_match_brute_force(seed: int, n: int, distinct: int) -> None:
    rng = np.random.default_rng(seed)
    score = rng.integers(0, distinct, n).astype(float)  # many ties on purpose
    treated = rng.random(n) < 0.5
    y = (rng.random(n) < 0.3).astype(float)
    xs, qs, us = _reference_curves(score, treated, y)
    q, u = qini_curve(score, treated, y), uplift_curve(score, treated, y)
    np.testing.assert_allclose(q.x, xs)
    np.testing.assert_allclose(q.y, qs, atol=1e-9)
    np.testing.assert_allclose(u.y, us, atol=1e-9)


def test_hand_computed_qini() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    treated = np.array([True, False, True, False])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    curve = qini_curve(score, treated, y)
    # after 1: 1 - 0; after 2: 1 - 0*1/1; after 3: 2 - 0*2/1; after 4: 2 - 1*2/2
    np.testing.assert_allclose(curve.y, [0.0, 1.0, 1.0, 2.0, 1.0])


def _randomized(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tau = rng.uniform(-0.1, 0.3, n)
    treated = rng.random(n) < 0.5
    y = (rng.random(n) < 0.2 + treated * tau).astype(float)
    return tau, treated, y


def test_oracle_beats_random_and_perfect_is_one() -> None:
    tau, treated, y = _randomized(20_000, 0)
    assert qini_coefficient(tau, treated, y) > 0.1
    assert auuc(tau, treated, y) > 0.1
    t = treated.astype(float)
    assert qini_coefficient(y * t - y * (1 - t), treated, y) == pytest.approx(1.0)


def test_random_scores_average_zero() -> None:
    _, treated, y = _randomized(4000, 1)
    rng = np.random.default_rng(2)
    values = [qini_coefficient(rng.random(len(y)), treated, y) for _ in range(60)]
    assert abs(np.mean(values)) < 3 * np.std(values) / np.sqrt(len(values)) + 0.01


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000))
def test_metrics_depend_only_on_the_ranking(seed: int) -> None:
    tau, treated, y = _randomized(500, seed)
    transforms: list[Callable[[np.ndarray], np.ndarray]] = [
        lambda s: 3 * s + 1,
        lambda s: np.exp(5 * s),
        lambda s: s**3,
    ]
    for transform in transforms:
        assert qini_coefficient(transform(tau), treated, y) == pytest.approx(
            qini_coefficient(tau, treated, y), abs=1e-12
        )
        assert uplift_at_k(transform(tau), treated, y, 0.3) == pytest.approx(
            uplift_at_k(tau, treated, y, 0.3), abs=1e-12
        )


def test_perfect_curve_dominates_any_ranking() -> None:
    tau, treated, y = _randomized(3000, 3)
    perfect = perfect_qini_curve(treated, y)
    model = qini_curve(tau, treated, y)
    grid = np.linspace(0, len(y), 50)
    assert np.all(np.interp(grid, perfect.x, perfect.y) >= np.interp(grid, model.x, model.y) - 1e-9)


def test_uplift_at_k_matches_manual_slice() -> None:
    tau, treated, y = _randomized(1000, 4)
    top = np.argsort(-tau, kind="stable")[:300]
    expected = y[top][treated[top]].mean() - y[top][~treated[top]].mean()
    assert uplift_at_k(tau, treated, y, 0.3) == pytest.approx(expected)
    with pytest.raises(ValueError, match="k must be"):
        uplift_at_k(tau, treated, y, 0.0)


def test_calibration_slope_matches_statsmodels_wls() -> None:
    tau, treated, y = _randomized(30_000, 5)
    calibration = calibration_by_bin(tau, treated, y, bins=10)
    assert [b.bin for b in calibration.bins] == list(range(1, 11))
    predicted = np.array([b.predicted for b in calibration.bins])
    observed = np.array([b.observed for b in calibration.bins])
    se = np.array([b.se for b in calibration.bins])
    wls = sm.WLS(observed, sm.add_constant(predicted), weights=1 / se**2).fit()
    assert calibration.slope == pytest.approx(wls.params[1], rel=1e-8)
    assert calibration.intercept == pytest.approx(wls.params[0], rel=1e-6, abs=1e-10)
    # The oracle score is the true uplift, so it is calibrated up to noise.
    assert calibration.slope == pytest.approx(1.0, abs=0.25)
