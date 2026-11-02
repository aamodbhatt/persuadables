"""Targeting policies as per-customer action probabilities.

A policy maps each customer to a distribution over arms (0 = no email, 1 = Mens, 2 = Womens).
Deterministic policies are one-hot rows; "email a random k%" is the stochastic row
``(1 - k, ..., k, ...)``. One representation lets one estimator value every policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import FloatArray, IntArray

NO_EMAIL = 0


@dataclass(frozen=True)
class Policy:
    name: str
    budget: float
    #: (n_customers, n_arms) action probabilities; rows sum to 1.
    probabilities: FloatArray

    def __post_init__(self) -> None:
        p = self.probabilities
        if p.ndim != 2 or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0):
            msg = (
                f"policy {self.name!r}: probabilities must be an (n, arms) matrix of distributions"
            )
            raise ValueError(msg)

    @property
    def emailed_share(self) -> float:
        """Expected share of customers who receive an email."""
        return float(1.0 - self.probabilities[:, NO_EMAIL].mean())


def _one_hot(arms: IntArray, n_arms: int) -> FloatArray:
    out = np.zeros((len(arms), n_arms))
    out[np.arange(len(arms)), arms] = 1.0
    return out


def email_no_one(n: int, n_arms: int) -> Policy:
    return Policy("no_one", 0.0, _one_hot(np.full(n, NO_EMAIL, dtype=np.int64), n_arms))


def email_everyone(n: int, n_arms: int, arm: int) -> Policy:
    return Policy(f"everyone_arm{arm}", 1.0, _one_hot(np.full(n, arm, dtype=np.int64), n_arms))


def email_random(n: int, n_arms: int, arm: int, budget: float) -> Policy:
    probabilities = np.zeros((n, n_arms))
    probabilities[:, NO_EMAIL] = 1.0 - budget
    probabilities[:, arm] += budget
    return Policy("random", budget, probabilities)


def email_top_k(
    name: str, scores: FloatArray, arms: IntArray, budget: float, n_arms: int
) -> Policy:
    """Email the ``round(budget * n)`` highest-scoring customers, each with their own arm.

    Ties are broken by row order (stable sort) so the policy is a deterministic function of the
    scores.
    """
    if not 0.0 <= budget <= 1.0:
        msg = f"budget must be in [0, 1], got {budget}"
        raise ValueError(msg)
    n_send = round(budget * len(scores))
    chosen = np.full(len(scores), NO_EMAIL, dtype=np.int64)
    top = np.argsort(-scores, kind="stable")[:n_send]
    chosen[top] = arms[top]
    return Policy(name, budget, _one_hot(chosen, n_arms))


def best_arm_by_uplift(
    uplift: FloatArray, treatment_arms: tuple[int, ...]
) -> tuple[FloatArray, IntArray]:
    """Per customer, the arm with the largest predicted uplift and that uplift."""
    best = np.argmax(uplift, axis=1)
    return uplift[np.arange(len(uplift)), best], np.asarray(treatment_arms, dtype=np.int64)[best]
