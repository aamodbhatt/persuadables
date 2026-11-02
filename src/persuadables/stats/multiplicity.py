"""Family-wise error control across arms, metrics and segments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from persuadables._types import FloatArray


def holm_adjust(p_values: Sequence[float] | FloatArray) -> FloatArray:
    """Holm-Bonferroni step-down adjusted p-values.

    Reject hypothesis i at level alpha when its adjusted p-value is <= alpha; this controls the
    family-wise error rate at alpha under any dependence between the tests, and is uniformly more
    powerful than plain Bonferroni.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        msg = "p_values must be one-dimensional"
        raise ValueError(msg)
    if np.any((p < 0) | (p > 1) | np.isnan(p)):
        msg = "p-values must lie in [0, 1]"
        raise ValueError(msg)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    stepped = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted
