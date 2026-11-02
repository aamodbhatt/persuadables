"""Off-policy value of a targeting policy from randomized (logged) data.

Customer i was randomly assigned arm A_i with known probability p(A_i) (one third each by design)
and produced outcome Y_i. For a policy pi, each customer is reweighted by
``w_i = pi(A_i | x_i) / p(A_i)``: customers whose random arm agrees with what the policy would
have done stand in for everyone the policy would have treated that way.

- IPW (Horvitz-Thompson): ``V = mean(w * Y)``, unbiased.
- SNIPW (Hajek): ``V = sum(w * Y) / sum(w)``, slightly biased, lower variance; the headline
  estimator here.

Uncertainty comes from a Poisson bootstrap over customers, shared across policies so that
differences between policies get paired (much tighter) intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from persuadables._types import FloatArray, IntArray
from persuadables.policy.policies import Policy

_BATCH = 100


def importance_weights(
    policy: Policy, logged_arm: IntArray, design_shares: FloatArray
) -> FloatArray:
    if len(logged_arm) != len(policy.probabilities):
        msg = "the policy and the logged data must cover the same customers"
        raise ValueError(msg)
    rows = np.arange(len(logged_arm))
    return np.asarray(policy.probabilities[rows, logged_arm] / design_shares[logged_arm])


@dataclass(frozen=True)
class BootstrapValues:
    #: (n_policies, n_outcomes) SNIPW point estimates.
    point: FloatArray
    #: (n_policies, n_outcomes) IPW point estimates (for the estimator cross-check).
    point_ipw: FloatArray
    #: (resamples, n_policies, n_outcomes) SNIPW estimates on Poisson-bootstrap resamples.
    draws: FloatArray


def snipw_bootstrap(
    weights: FloatArray,
    outcomes: FloatArray,
    *,
    resamples: int,
    seed: int,
    labels: Sequence[str] | None = None,
) -> BootstrapValues:
    """SNIPW values of many policies (columns of ``weights``) on many outcomes (columns of
    ``outcomes``), with one shared set of bootstrap resamples.

    The Poisson(1) bootstrap draws an independent count per customer instead of a multinomial
    resample; for n in the thousands the two are equivalent and Poisson vectorizes as a single
    matrix product.

    Raises ``ValueError`` (naming the policy from ``labels`` when given) if a policy has zero
    total weight: no logged customer received any action it takes, so its value is unidentified.
    A resample in which that happens by chance yields NaN for that draw and is skipped by the
    quantile-based intervals.
    """
    n, n_policies = weights.shape
    if outcomes.shape[0] != n:
        msg = "weights and outcomes must have one row per customer"
        raise ValueError(msg)
    if labels is not None and len(labels) != n_policies:
        msg = f"got {len(labels)} labels for {n_policies} policies"
        raise ValueError(msg)
    totals = weights.sum(axis=0)
    unidentified = [int(c) for c in np.flatnonzero(totals <= 0)]
    if unidentified:
        names = [labels[c] if labels is not None else f"column {c}" for c in unidentified]
        msg = (
            f"no logged customer received an action taken by {', '.join(names)}; "
            "the value of such a policy is unidentified from this data"
        )
        raise ValueError(msg)
    n_outcomes = outcomes.shape[1]
    weighted = (weights[:, :, None] * outcomes[:, None, :]).reshape(n, n_policies * n_outcomes)
    point = (weighted.sum(axis=0) / np.repeat(totals, n_outcomes)).reshape(n_policies, n_outcomes)
    point_ipw = (weighted.sum(axis=0) / n).reshape(n_policies, n_outcomes)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, n_policies, n_outcomes))
    for start in range(0, resamples, _BATCH):
        stop = min(start + _BATCH, resamples)
        counts = rng.poisson(1.0, size=(stop - start, n)).astype(np.float64)
        numerators = (counts @ weighted).reshape(stop - start, n_policies, n_outcomes)
        denominators = (counts @ weights)[:, :, None]
        draws[start:stop] = numerators / np.where(denominators > 0, denominators, np.nan)
    return BootstrapValues(point=point, point_ipw=point_ipw, draws=draws)
