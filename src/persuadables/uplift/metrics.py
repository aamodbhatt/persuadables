"""Ranking metrics for uplift models, evaluated on randomized data.

An uplift model cannot be scored per customer (nobody is observed both emailed and not), so it is
scored on how well its ranking sorts customers by effect. Sort by predicted uplift, descending;
after targeting the top ``n`` customers, with ``n_t``/``n_c`` treated/control among them and
``y_t``/``y_c`` their outcome totals:

- Qini curve:   ``Q(n) = y_t - y_c * n_t / n_c`` (incremental outcomes, control rescaled to the
  treated count; Radcliffe, 2007).
- Uplift curve: ``U(n) = (y_t / n_t - y_c / n_c) * n`` (incremental outcomes if all n were
  treated; Rzepakowski and Jaroszewicz, 2012).

Tied scores are cut together (a ranking cannot split a tie), and the curve definitions, the
"perfect" reference curves and the normalized areas follow scikit-uplift, so the numbers are
comparable with published Hillstrom results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from persuadables._types import BoolArray, FloatArray


@dataclass(frozen=True)
class Curve:
    """A gain curve: ``x`` customers targeted (starting at 0) against the curve value ``y``."""

    x: FloatArray
    y: FloatArray


def _cumulative(
    score: FloatArray, treated: BoolArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if not len(score) == len(treated) == len(y) or len(y) == 0:
        msg = "score, treated and y must be non-empty and of equal length"
        raise ValueError(msg)
    order = np.argsort(-score, kind="stable")
    s, t, yy = score[order], treated[order].astype(np.float64), y[order]
    cut = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    n_all = (cut + 1).astype(np.float64)
    n_t = np.cumsum(t)[cut]
    y_t = np.cumsum(yy * t)[cut]
    y_c = np.cumsum(yy * (1 - t))[cut]
    return n_all, n_t, n_all - n_t, y_t, y_c


def _ratio(num: FloatArray, den: FloatArray) -> FloatArray:
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def _with_origin(x: FloatArray, y: FloatArray) -> Curve:
    return Curve(np.r_[0.0, x], np.r_[0.0, y])


def qini_curve(score: FloatArray, treated: BoolArray, y: FloatArray) -> Curve:
    n_all, n_t, n_c, y_t, y_c = _cumulative(score, treated, y)
    return _with_origin(n_all, y_t - y_c * _ratio(n_t, n_c))


def uplift_curve(score: FloatArray, treated: BoolArray, y: FloatArray) -> Curve:
    n_all, n_t, n_c, y_t, y_c = _cumulative(score, treated, y)
    return _with_origin(n_all, (_ratio(y_t, n_t) - _ratio(y_c, n_c)) * n_all)


def perfect_qini_curve(treated: BoolArray, y: FloatArray) -> Curve:
    """The Qini curve of an oracle that ranks treated responders first and control responders last
    (allowing for negative effects)."""
    t = treated.astype(np.float64)
    return qini_curve(y * t - y * (1 - t), treated, y)


def perfect_uplift_curve(treated: BoolArray, y: FloatArray) -> Curve:
    t = treated.astype(np.float64)
    control_responders = np.sum((y == 1) & ~treated)
    treated_non_responders = np.sum((y == 0) & treated)
    summand = y if control_responders > treated_non_responders else t
    return uplift_curve(2.0 * (y == t) + summand, treated, y)


def _area(curve: Curve) -> float:
    return float(np.trapezoid(curve.y, curve.x))


def _normalized_area(model: Curve, perfect: Curve) -> float:
    baseline = Curve(np.array([0.0, perfect.x[-1]]), np.array([0.0, perfect.y[-1]]))
    denominator = _area(perfect) - _area(baseline)
    if denominator == 0:
        return math.nan
    return (_area(model) - _area(baseline)) / denominator


def qini_coefficient(score: FloatArray, treated: BoolArray, y: FloatArray) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal, as a share of the
    same area for the perfect curve. 0 is random targeting, 1 is the oracle; it can be negative."""
    return _normalized_area(qini_curve(score, treated, y), perfect_qini_curve(treated, y))


def auuc(score: FloatArray, treated: BoolArray, y: FloatArray) -> float:
    """Normalized area under the uplift curve, on the same 0 (random) to 1 (oracle) scale."""
    return _normalized_area(uplift_curve(score, treated, y), perfect_uplift_curve(treated, y))


def uplift_at_k(score: FloatArray, treated: BoolArray, y: FloatArray, k: float) -> float:
    """Treated minus control outcome rate among the top ``k`` share of customers by score.

    NaN when the top slice holds no treated or no control customer.
    """
    if not 0 < k <= 1:
        msg = f"k must be in (0, 1], got {k}"
        raise ValueError(msg)
    n_top = int(len(score) * k)
    if n_top == 0:
        return math.nan
    top = np.argsort(-score, kind="stable")[:n_top]
    t, yy = treated[top], y[top]
    if t.all() or not t.any():
        return math.nan
    return float(yy[t].mean() - yy[~t].mean())


@dataclass(frozen=True)
class CalibrationBin:
    """One quantile bin of predicted uplift (bin 1 holds the highest predictions)."""

    bin: int
    n_treatment: int
    n_control: int
    predicted: float
    observed: float
    se: float


@dataclass(frozen=True)
class Calibration:
    bins: list[CalibrationBin]
    #: Inverse-variance weighted least-squares fit of observed on predicted uplift across bins.
    slope: float
    intercept: float


def calibration_by_bin(
    predicted: FloatArray, treated: BoolArray, y: FloatArray, *, bins: int
) -> Calibration:
    """Observed uplift (difference in means) against mean predicted uplift in equal-size bins.

    A calibrated model has observed = predicted in every bin: slope 1, intercept 0. A model can
    rank well (high Qini) and still be miscalibrated, which matters as soon as predicted uplift is
    multiplied by a margin to make a send/no-send decision.
    """
    if len(predicted) < bins:
        msg = f"need at least {bins} rows for {bins} bins, got {len(predicted)}"
        raise ValueError(msg)
    rank = np.empty(len(predicted), dtype=np.int64)
    rank[np.argsort(-predicted, kind="stable")] = np.arange(len(predicted))
    bin_of = rank * bins // len(predicted)
    rows: list[CalibrationBin] = []
    for b in range(bins):
        member = bin_of == b
        y_t, y_c = y[member & treated], y[member & ~treated]
        if len(y_t) < 2 or len(y_c) < 2:
            continue
        se = math.sqrt(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c, ddof=1) / len(y_c))
        rows.append(
            CalibrationBin(
                bin=b + 1,
                n_treatment=len(y_t),
                n_control=len(y_c),
                predicted=float(predicted[member].mean()),
                observed=float(y_t.mean() - y_c.mean()),
                se=se,
            )
        )
    slope, intercept = _weighted_line(
        np.array([r.predicted for r in rows]),
        np.array([r.observed for r in rows]),
        np.array([r.se for r in rows]),
    )
    return Calibration(rows, slope, intercept)


def _weighted_line(x: FloatArray, y: FloatArray, se: FloatArray) -> tuple[float, float]:
    usable = se > 0
    if usable.sum() < 2 or np.ptp(x[usable]) == 0:
        return math.nan, math.nan
    w = 1.0 / se[usable] ** 2
    design = np.column_stack([np.ones(int(usable.sum())), x[usable]])
    sqrt_w = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(design * sqrt_w, y[usable] * sqrt_w[:, 0], rcond=None)
    return float(coef[1]), float(coef[0])
