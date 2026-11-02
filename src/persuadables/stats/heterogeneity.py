"""Heterogeneity of a treatment effect across the levels of a pre-registered segment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from persuadables._types import FloatArray


@dataclass(frozen=True)
class CochranQ:
    q: float
    df: int
    p_value: float
    #: Levels that entered the test (levels with a zero standard error carry no weight).
    levels_used: int


def cochran_q(effects: FloatArray, standard_errors: FloatArray) -> CochranQ:
    """Cochran's Q test that every level shares one effect.

    ``Q = sum w_i (theta_i - theta_bar)^2`` with inverse-variance weights ``w_i = 1/se_i^2`` and
    ``theta_bar`` the fixed-effect mean; under the null Q ~ chi2(k - 1). The levels of a segment are
    disjoint groups of customers, so their estimates are independent, which the test needs.
    """
    if effects.shape != standard_errors.shape or effects.ndim != 1:
        msg = "effects and standard_errors must be 1-D arrays of the same length"
        raise ValueError(msg)
    usable = standard_errors > 0
    k = int(usable.sum())
    if k < 2:
        return CochranQ(q=0.0, df=0, p_value=1.0, levels_used=k)
    weights = 1.0 / standard_errors[usable] ** 2
    theta = effects[usable]
    pooled = float(np.sum(weights * theta) / np.sum(weights))
    q = float(np.sum(weights * (theta - pooled) ** 2))
    return CochranQ(q=q, df=k - 1, p_value=float(stats.chi2.sf(q, k - 1)), levels_used=k)
