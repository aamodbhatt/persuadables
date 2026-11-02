"""Power, minimum detectable effect and sample size for a two-sample comparison of means.

Normal approximation throughout (the arms here hold ~21,000 customers each). For a proportion,
pass ``sigma = sqrt(p * (1 - p))`` at the baseline rate.
"""

from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1 or not 0 < power < 1:
        msg = f"alpha and power must lie in (0, 1), got alpha={alpha}, power={power}"
        raise ValueError(msg)
    return float(stats.norm.isf(alpha / 2)), float(stats.norm.ppf(power))


def minimum_detectable_effect(
    sigma: float, n_treatment: int, n_control: int, *, alpha: float, power: float
) -> float:
    """Smallest absolute effect a two-sided level-alpha test detects with the given power.

    ``(z_{1-alpha/2} + z_{power}) * sigma * sqrt(1/n_t + 1/n_c)``; the far tail's contribution to
    power (below 1e-5 at 80% power) is ignored, as is standard.
    """
    z_alpha, z_power = _z(alpha, power)
    return (z_alpha + z_power) * sigma * math.sqrt(1 / n_treatment + 1 / n_control)


def power_two_sample(
    effect: float, sigma: float, n_treatment: int, n_control: int, *, alpha: float
) -> float:
    """Probability that a two-sided level-alpha z-test rejects at true effect ``effect``."""
    z_alpha = float(stats.norm.isf(alpha / 2))
    shift = abs(effect) / (sigma * math.sqrt(1 / n_treatment + 1 / n_control))
    return float(stats.norm.sf(z_alpha - shift) + stats.norm.cdf(-z_alpha - shift))


def sample_size_per_arm(effect: float, sigma: float, *, alpha: float, power: float) -> int:
    """Customers needed in EACH of two equal arms to detect ``effect`` with the given power."""
    if effect == 0:
        msg = "a zero effect cannot be detected at any sample size"
        raise ValueError(msg)
    z_alpha, z_power = _z(alpha, power)
    return math.ceil(2 * ((z_alpha + z_power) * sigma / effect) ** 2)
