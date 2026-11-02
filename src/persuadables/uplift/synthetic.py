"""A randomized experiment with a KNOWN conditional treatment effect, to test that the learners
recover heterogeneity rather than just fit outcomes.

The design separates "likely to respond" from "persuadable" on purpose: the baseline response
depends on x0 and x1, while the treatment effect is driven by x2 (and the binary x4) and is
negative for part of the population ("sleeping dogs"). A response model ranks by x0/x1 and
targets the wrong people; a working uplift model ranks by x2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import BoolArray, FloatArray

FEATURE_NAMES: tuple[str, ...] = ("x0", "x1", "x2", "x3", "x4")


def _sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


@dataclass(frozen=True)
class SyntheticExperiment:
    X: FloatArray
    treated: BoolArray
    y: FloatArray
    #: True P(y=1 | x, control) and P(y=1 | x, treated).
    mu0: FloatArray
    mu1: FloatArray

    @property
    def tau(self) -> FloatArray:
        """The true uplift of every customer."""
        return self.mu1 - self.mu0


def true_uplift(X: FloatArray) -> FloatArray:
    """tau(x): from about -0.08 (x2 low) to about +0.30 (x2 high, x4 = 1)."""
    return 0.30 * _sigmoid(3.0 * X[:, 2]) - 0.08 + 0.08 * X[:, 4]


def baseline_response(X: FloatArray) -> FloatArray:
    return _sigmoid(-1.2 + 1.2 * X[:, 0] + 0.8 * X[:, 1] - 0.4 * X[:, 3])


def make_synthetic_experiment(
    n: int, *, seed: int, treated_share: float = 0.5
) -> SyntheticExperiment:
    rng = np.random.default_rng(seed)
    continuous = rng.normal(size=(n, 4))
    binary = rng.integers(0, 2, size=(n, 1)).astype(np.float64)
    X = np.column_stack([continuous, binary])
    mu0 = baseline_response(X)
    mu1 = np.clip(mu0 + true_uplift(X), 0.001, 0.999)
    treated = rng.random(n) < treated_share
    y = (rng.random(n) < np.where(treated, mu1, mu0)).astype(np.float64)
    return SyntheticExperiment(X=X, treated=treated, y=y, mu0=mu0, mu1=mu1)
