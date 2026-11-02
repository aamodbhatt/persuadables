"""Checks on the randomization itself: sample-ratio mismatch and covariate balance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from persuadables._types import FloatArray, IntArray


@dataclass(frozen=True)
class SampleRatioResult:
    counts: tuple[int, ...]
    expected_counts: tuple[float, ...]
    chi2: float
    df: int
    p_value: float


def sample_ratio_mismatch(
    counts: tuple[int, ...], expected_shares: tuple[float, ...]
) -> SampleRatioResult:
    """Pearson chi-square goodness-of-fit of observed arm sizes to the designed allocation.

    A tiny p-value means the randomizer or the logging is broken (assignment depends on something),
    and every downstream estimate is suspect, however significant it looks.
    """
    if len(counts) != len(expected_shares) or len(counts) < 2:
        msg = "counts and expected_shares must have the same length (at least 2)"
        raise ValueError(msg)
    if not math.isclose(sum(expected_shares), 1.0, abs_tol=1e-9):
        msg = f"expected shares must sum to 1, got {sum(expected_shares)}"
        raise ValueError(msg)
    total = sum(counts)
    expected = [share * total for share in expected_shares]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(counts, expected, strict=True))
    df = len(counts) - 1
    return SampleRatioResult(
        counts=tuple(counts),
        expected_counts=tuple(expected),
        chi2=chi2,
        df=df,
        p_value=float(stats.chi2.sf(chi2, df)),
    )


def standardized_mean_difference(treatment: FloatArray, control: FloatArray) -> float:
    """(mean_t - mean_c) / sqrt((var_t + var_c) / 2), the scale-free balance metric.

    Unlike a t-test it does not grow with sample size, so it measures how different the arms are
    rather than how sure we are that they differ. |SMD| < 0.1 is the conventional balance bar.
    """
    diff = float(np.mean(treatment) - np.mean(control))
    pooled_sd = math.sqrt((float(np.var(treatment, ddof=1)) + float(np.var(control, ddof=1))) / 2)
    if pooled_sd == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return diff / pooled_sd


@dataclass(frozen=True)
class BalanceRow:
    covariate: str
    arm: int
    smd: float


def balance_table(
    arm: IntArray,
    X: FloatArray,
    names: tuple[str, ...],
    *,
    control: int,
    treatments: tuple[int, ...],
) -> list[BalanceRow]:
    """SMD of every covariate (column of X) for every treatment arm against the control arm."""
    if X.shape != (len(arm), len(names)):
        msg = f"X must be (n, {len(names)}) to match names; got {X.shape}"
        raise ValueError(msg)
    is_control = arm == control
    return [
        BalanceRow(
            name, treatment, standardized_mean_difference(X[arm == treatment, j], X[is_control, j])
        )
        for treatment in treatments
        for j, name in enumerate(names)
    ]
