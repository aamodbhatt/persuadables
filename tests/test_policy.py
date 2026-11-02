"""Off-policy evaluation: IPW recovers known policy values on a simulated three-arm experiment,
and the policy comparison reports planted per-1,000-email values with the right signs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from persuadables.config import OUTCOMES, Config, load_config
from persuadables.policy.evaluation import PolicyEvaluation, evaluate_policies
from persuadables.policy.ope import importance_weights, snipw_bootstrap
from persuadables.policy.policies import (
    Policy,
    best_arm_by_uplift,
    email_everyone,
    email_no_one,
    email_random,
    email_top_k,
)
from persuadables.training import prediction_key
from tests.fakes import ROOT

DESIGN = np.full(3, 1 / 3)


def _three_arm(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Potential outcome probabilities per arm, a random assignment, and observed outcomes."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    mu = np.column_stack([0.10 + 0 * x, 0.14 + 0.08 * x, 0.12 - 0.06 * x])
    arm = rng.integers(0, 3, n)
    y = (rng.random(n) < mu[np.arange(n), arm]).astype(float)
    return x, mu, arm, y


def _reference_ipw(weights: np.ndarray, y: np.ndarray) -> float:
    """Horvitz-Thompson, written out as the textbook formula."""
    return float(np.mean(weights * y))


def _reference_snipw(weights: np.ndarray, y: np.ndarray) -> float:
    """Hajek, written out as the textbook formula."""
    return float(np.sum(weights * y) / np.sum(weights))


def _values(policy: Policy, arm: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(SNIPW, IPW) value of one policy on one outcome, from the package estimator."""
    weights = importance_weights(policy, arm, DESIGN)
    boot = snipw_bootstrap(weights[:, None], y[:, None], resamples=1, seed=0)
    return float(boot.point[0, 0]), float(boot.point_ipw[0, 0])


def test_ipw_recovers_the_value_of_a_targeting_policy() -> None:
    x, mu, arm, y = _three_arm(300_000, 0)
    chosen = np.where(x > 0, 1, 2)  # email arm 1 to x > 0, arm 2 to the rest
    policy = email_top_k("rule", np.ones(len(x)), chosen, 1.0, 3)
    true_value = float(mu[np.arange(len(x)), chosen].mean())
    snipw, ipw = _values(policy, arm, y)
    assert ipw == pytest.approx(true_value, abs=0.003)
    assert snipw == pytest.approx(true_value, abs=0.003)


def test_constant_policy_snipw_is_the_arm_mean() -> None:
    _, _, arm, y = _three_arm(5000, 1)
    snipw, _ = _values(email_everyone(len(arm), 3, 2), arm, y)
    assert snipw == pytest.approx(y[arm == 2].mean(), rel=1e-12)


def test_random_policy_ipw_is_linear_in_the_budget() -> None:
    _, _, arm, y = _three_arm(5000, 2)
    n = len(arm)

    def value(p: Policy) -> float:
        return _values(p, arm, y)[1]

    for budget in (0.0, 0.25, 0.7, 1.0):
        expected = (1 - budget) * value(email_no_one(n, 3)) + budget * value(
            email_everyone(n, 3, 1)
        )
        assert value(email_random(n, 3, 1, budget)) == pytest.approx(expected, rel=1e-12)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 400), st.floats(0, 1), st.integers(0, 10_000))
def test_top_k_emails_exactly_the_budget(n: int, budget: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n).round(1)  # ties
    arms = rng.integers(1, 3, n)
    policy = email_top_k("p", scores, arms, budget, 3)
    emailed = policy.probabilities[:, 0] == 0
    assert emailed.sum() == round(budget * n)
    if 0 < emailed.sum() < n:
        assert scores[emailed].min() >= scores[~emailed].max()
    np.testing.assert_array_equal(policy.probabilities[emailed].argmax(axis=1), arms[emailed])


def test_best_arm_by_uplift_picks_the_larger_uplift() -> None:
    uplift = np.array([[0.1, 0.3], [0.2, -0.1], [-0.2, -0.1]])
    score, arm = best_arm_by_uplift(uplift, (1, 2))
    np.testing.assert_allclose(score, [0.3, 0.2, -0.1])
    np.testing.assert_array_equal(arm, [2, 1, 2])


def test_policy_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="distributions"):
        Policy("bad", 0.5, np.array([[0.5, 0.6, 0.0]]))


def test_bootstrap_point_matches_direct_estimators_and_brackets_it() -> None:
    _, _, arm, y = _three_arm(6000, 3)
    n = len(arm)
    policies = [email_no_one(n, 3), email_everyone(n, 3, 1), email_random(n, 3, 2, 0.4)]
    weights = np.column_stack([importance_weights(p, arm, DESIGN) for p in policies])
    outcomes = np.column_stack([y, 2 * y])
    boot = snipw_bootstrap(weights, outcomes, resamples=300, seed=0)
    for c in range(len(policies)):
        assert boot.point[c, 0] == pytest.approx(_reference_snipw(weights[:, c], y), rel=1e-12)
        assert boot.point_ipw[c, 0] == pytest.approx(_reference_ipw(weights[:, c], y), rel=1e-12)
        assert boot.point[c, 1] == pytest.approx(2 * boot.point[c, 0], rel=1e-12)
        low, high = np.quantile(boot.draws[:, c, 0], [0.025, 0.975])
        assert low < boot.point[c, 0] < high
    assert boot.draws.shape == (300, 3, 2)


def test_an_unidentified_policy_is_an_error_that_names_it() -> None:
    arm = np.array([0, 0, 1, 1])  # nobody was logged with the Womens email
    y = np.array([0.0, 1.0, 1.0, 0.0])
    policies = [email_no_one(4, 3), email_everyone(4, 3, 2)]
    weights = np.column_stack([importance_weights(p, arm, DESIGN) for p in policies])
    with pytest.raises(ValueError, match="everyone_arm2"):
        snipw_bootstrap(
            weights, y[:, None], resamples=10, seed=0, labels=[p.name for p in policies]
        )


# evaluate_policies on a planted experiment whose policy values are known exactly.

_N = 6000  # 1,000 customers in every (group, arm) cell
#: Visit rate per group and arm (no email, Mens, Womens). Group A is persuaded by the Mens email.
_VISIT_A = np.array([0.10, 0.40, 0.20])
_VISIT_B = np.array([0.10, 0.10, 0.15])


def _planted() -> tuple[dict[str, np.ndarray], Config]:
    i = np.arange(_N)
    arm = i % 3
    in_a = (i // 3) % 2 == 0  # each group holds every arm equally often
    # Deterministic per (group, arm) cell, so every SNIPW point estimate is exact.
    visit = np.where(in_a, _VISIT_A[arm], _VISIT_B[arm])
    test: dict[str, np.ndarray] = {
        "arm": arm.astype(np.float64),
        "visit": visit,
        "conversion": 0.1 * visit,
        "spend": 10.0 * visit,
        # The uplift models know the truth; the response model ranks group B first.
        prediction_key("uplift", "visit", 1, "s"): np.where(in_a, 0.30, 0.00),
        prediction_key("uplift", "visit", 2, "s"): np.where(in_a, 0.10, 0.05),
        prediction_key("response", "visit", 1): np.where(in_a, 0.0, 1.0),
    }
    base = load_config(ROOT / "configs" / "default.toml", root=ROOT)
    policy = base.policy.model_copy(
        update={
            "rank_outcome": "visit",
            "headline_budget": 0.5,
            "budgets": (0.5,),
            "grid_step": 0.5,
            "bootstrap_resamples": 50,
        }
    )
    return test, base.model_copy(update={"policy": policy})


def _evaluate_planted() -> PolicyEvaluation:
    test, config = _planted()
    return evaluate_policies(test, {"visit": {"1": "s", "2": "s"}}, {"visit": 1}, config)


def test_policy_values_per_1000_emails_match_the_planted_effects() -> None:
    result = _evaluate_planted()
    # Incremental visits over emailing no one, per 1,000 emails sent.
    expected = {
        ("everyone_arm1", 1.0): 150.0,  # (0.25 - 0.10) * 1000
        ("everyone_arm2", 1.0): 75.0,  # (0.175 - 0.10) * 1000
        ("uplift_multi", 0.5): 300.0,  # all of group A, Mens email: +0.30 each
        ("uplift_single", 0.5): 300.0,
        ("response", 0.5): 0.0,  # all of group B, Mens email: no effect
        ("random", 0.5): 150.0,  # the same per email as emailing everyone
    }
    scale = {"visit": 1.0, "conversion": 0.1, "spend": 10.0}
    rows = {(r.policy, r.budget): r for r in result.table}
    for key, visits in expected.items():
        row = rows[key]
        assert row.emailed_share == pytest.approx(key[1])
        for value in row.values:
            incremental = value.incremental_per_1000_emails
            assert incremental.estimate == pytest.approx(visits * scale[value.outcome], abs=1e-9)
            assert incremental.low <= incremental.estimate + 1e-9
            assert incremental.estimate - 1e-9 <= incremental.high
    no_one = rows[("no_one", 0.0)]
    assert all(math.isnan(v.incremental_per_1000_emails.estimate) for v in no_one.values)
    assert [v.outcome for v in no_one.values] == list(OUTCOMES)


def test_policy_comparisons_are_uplift_minus_baseline() -> None:
    result = _evaluate_planted()
    differences = {
        (c.policy_b, c.outcome): c.difference_per_1000_emails.estimate for c in result.comparisons
    }
    assert {c.policy_a for c in result.comparisons} == {"uplift_multi"}
    assert differences[("random", "visit")] == pytest.approx(150.0, abs=1e-9)
    assert differences[("response", "visit")] == pytest.approx(300.0, abs=1e-9)
    assert differences[("uplift_single", "visit")] == pytest.approx(0.0, abs=1e-9)
    assert differences[("response", "conversion")] == pytest.approx(30.0, abs=1e-9)
    assert differences[("random", "spend")] == pytest.approx(1500.0, abs=1e-9)
