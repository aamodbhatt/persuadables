"""CUPED: variance reduction with pre-period covariates (Deng, Xu, Kohavi and Walker, 2013).

For an outcome y and pre-experiment covariates X, the adjusted outcome is
``y_adj = y - (X - mean(X)) @ theta`` with theta the least-squares slope of y on X over the pooled
sample. Because X was fixed before randomization, ``E[X]`` is the same in both arms and the
difference in means of ``y_adj`` is still unbiased for the treatment effect; its variance shrinks
by the share of variance in y that X explains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import BoolArray, FloatArray
from persuadables.stats.inference import EffectEstimate, welch_ttest


@dataclass(frozen=True)
class CupedResult:
    raw: EffectEstimate
    adjusted: EffectEstimate
    theta: tuple[float, ...]
    #: 1 - Var(y_adj) / Var(y) over the pooled sample: the share of variance removed.
    variance_reduction: float


def cuped_theta(y: FloatArray, X: FloatArray) -> FloatArray:
    """Least-squares slope(s) of y on the centred covariates (multivariate CUPED)."""
    if X.ndim != 2 or len(X) != len(y):
        msg = f"X must be (n, k) with n = len(y); got {X.shape} and {len(y)}"
        raise ValueError(msg)
    centred = X - X.mean(axis=0)
    theta, *_ = np.linalg.lstsq(centred, y - y.mean(), rcond=None)
    return np.asarray(theta, dtype=np.float64)


def cuped_adjust(y: FloatArray, X: FloatArray, theta: FloatArray) -> FloatArray:
    return np.asarray(y - (X - X.mean(axis=0)) @ theta, dtype=np.float64)


def cuped_effect(
    y: FloatArray, X: FloatArray, treated: BoolArray, *, alpha: float = 0.05
) -> CupedResult:
    """Raw and CUPED-adjusted Welch estimates of the treatment effect on y."""
    theta = cuped_theta(y, X)
    adjusted = cuped_adjust(y, X, theta)
    raw_var = float(np.var(y, ddof=1))
    reduction = 1.0 - float(np.var(adjusted, ddof=1)) / raw_var if raw_var > 0 else 0.0
    return CupedResult(
        raw=welch_ttest(y[treated], y[~treated], alpha=alpha),
        adjusted=welch_ttest(adjusted[treated], adjusted[~treated], alpha=alpha),
        theta=tuple(float(v) for v in theta),
        variance_reduction=reduction,
    )
