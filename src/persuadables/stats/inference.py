"""Two-sample effect estimates: the two-proportion z-test, Welch's t-test and a stratified
percentile bootstrap for the difference in means.

Conventions: ``diff = mean(treatment) - mean(control)``; tests are two-sided; confidence intervals
have level ``1 - alpha``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from persuadables._types import FloatArray

#: Bootstrap resamples drawn per vectorized batch: bounds memory at ~batch * n * 4 bytes.
_BOOTSTRAP_BATCH = 200


@dataclass(frozen=True)
class EffectEstimate:
    """A difference in means between a treatment and a control sample."""

    method: str
    n_treatment: int
    n_control: int
    treatment_mean: float
    control_mean: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float


@dataclass(frozen=True)
class BootstrapInterval:
    low: float
    high: float
    se: float
    resamples: int


def two_proportion_ztest(
    successes_treatment: int,
    n_treatment: int,
    successes_control: int,
    n_control: int,
    *,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Difference of two proportions.

    The test statistic uses the pooled standard error (the null hypothesis says both arms share one
    rate); the confidence interval uses the unpooled (Wald) standard error, which does not assume
    the null. Mixing them is standard practice and matches statsmodels' ``proportions_ztest`` and
    ``confint_proportions_2indep(method="wald")``.
    """
    _require_counts(successes_treatment, n_treatment)
    _require_counts(successes_control, n_control)
    p_t = successes_treatment / n_treatment
    p_c = successes_control / n_control
    diff = p_t - p_c
    se = math.sqrt(p_t * (1 - p_t) / n_treatment + p_c * (1 - p_c) / n_control)
    pooled = (successes_treatment + successes_control) / (n_treatment + n_control)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n_treatment + 1 / n_control))
    # With no events (or only events) in both arms there is no evidence of a difference.
    z = diff / se_null if se_null > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    z_crit = float(stats.norm.isf(alpha / 2))
    return EffectEstimate(
        method="two-proportion z-test",
        n_treatment=n_treatment,
        n_control=n_control,
        treatment_mean=p_t,
        control_mean=p_c,
        diff=diff,
        se=se,
        ci_low=diff - z_crit * se,
        ci_high=diff + z_crit * se,
        statistic=z,
        p_value=p_value,
    )


def welch_ttest(
    treatment: FloatArray, control: FloatArray, *, alpha: float = 0.05
) -> EffectEstimate:
    """Welch's unequal-variance t-test with the Welch-Satterthwaite degrees of freedom."""
    n_t, n_c = len(treatment), len(control)
    if n_t < 2 or n_c < 2:
        msg = f"Welch's t-test needs at least 2 observations per arm, got {n_t} and {n_c}"
        raise ValueError(msg)
    mean_t, mean_c = float(np.mean(treatment)), float(np.mean(control))
    v_t = float(np.var(treatment, ddof=1)) / n_t
    v_c = float(np.var(control, ddof=1)) / n_c
    se = math.sqrt(v_t + v_c)
    diff = mean_t - mean_c
    if se == 0:
        return EffectEstimate(
            "Welch t-test", n_t, n_c, mean_t, mean_c, diff, 0.0, diff, diff, 0.0, 1.0
        )
    df = (v_t + v_c) ** 2 / (v_t**2 / (n_t - 1) + v_c**2 / (n_c - 1))
    t = diff / se
    t_crit = float(stats.t.isf(alpha / 2, df))
    return EffectEstimate(
        method="Welch t-test",
        n_treatment=n_t,
        n_control=n_c,
        treatment_mean=mean_t,
        control_mean=mean_c,
        diff=diff,
        se=se,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        statistic=t,
        p_value=float(2 * stats.t.sf(abs(t), df)),
    )


def bootstrap_mean_diff(
    treatment: FloatArray,
    control: FloatArray,
    *,
    resamples: int,
    alpha: float = 0.05,
    seed: int,
) -> BootstrapInterval:
    """Percentile bootstrap interval for ``mean(treatment) - mean(control)``.

    Each arm is resampled separately (stratified), which mirrors the design: arm sizes were fixed
    by the randomizer, not drawn from the population. Used for spend, whose zero-inflated,
    heavy-tailed distribution makes the t-interval's normal approximation the thing to check.
    """
    if resamples < 1:
        msg = f"resamples must be positive, got {resamples}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    draws = _bootstrap_means(treatment, resamples, rng) - _bootstrap_means(control, resamples, rng)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return BootstrapInterval(float(low), float(high), float(np.std(draws, ddof=1)), resamples)


def _bootstrap_means(values: FloatArray, resamples: int, rng: np.random.Generator) -> FloatArray:
    n = len(values)
    if n == 0:
        msg = "cannot bootstrap an empty sample"
        raise ValueError(msg)
    means = np.empty(resamples)
    for start in range(0, resamples, _BOOTSTRAP_BATCH):
        stop = min(start + _BOOTSTRAP_BATCH, resamples)
        index = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        means[start:stop] = values[index].mean(axis=1)
    return means


def _require_counts(successes: int, n: int) -> None:
    if n <= 0 or not 0 <= successes <= n:
        msg = f"need 0 <= successes <= n and n > 0, got successes={successes}, n={n}"
        raise ValueError(msg)
