"""Value every targeting policy on the held-out randomized test split.

Compared at every budget (share of customers emailed):

- ``uplift_multi``: the uplift models pick, per customer, the email with the larger predicted
  uplift, and the budget goes to the customers with the largest such uplift.
- ``uplift_single``: the same ranking, but everyone targeted gets the single best email.
- ``response``: the classic approach: email the customers with the highest predicted
  P(ranking outcome | best single email), a response model on the SAME outcome the uplift
  policies rank by (visit by default, so "likely visitors", not "likely buyers"), ignoring
  whether they would have responded anyway. Everyone targeted gets the best single email.
- ``random``: email a random share of customers.

plus the fixed policies "email everyone" (each email) and "email no one".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import FloatArray, IntArray
from persuadables.config import OUTCOMES, Config, LearnerName, Outcome
from persuadables.data.frame import ARMS, TREATMENT_ARMS
from persuadables.io import derive_seed
from persuadables.policy.ope import BootstrapValues, importance_weights, snipw_bootstrap
from persuadables.policy.policies import (
    Policy,
    best_arm_by_uplift,
    email_everyone,
    email_no_one,
    email_random,
    email_top_k,
)
from persuadables.training import prediction_key

BUDGETED_POLICIES: tuple[str, ...] = ("uplift_multi", "uplift_single", "response", "random")


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class OutcomeValue:
    outcome: Outcome
    #: Expected outcome per customer under the policy (SNIPW), with a bootstrap interval.
    per_customer: Interval
    #: The unnormalized IPW estimate of the same value, as an estimator cross-check.
    per_customer_ipw: float
    #: Extra outcomes over emailing no one, per 1,000 emails sent (NaN if nobody is emailed).
    incremental_per_1000_emails: Interval


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    budget: float
    emailed_share: float
    values: list[OutcomeValue]


@dataclass(frozen=True)
class Comparison:
    """Paired bootstrap difference in incremental outcomes per 1,000 emails: a minus b."""

    budget: float
    outcome: Outcome
    policy_a: str
    policy_b: str
    difference_per_1000_emails: Interval


@dataclass(frozen=True)
class PolicyCurve:
    policy: str
    outcome: Outcome
    budgets: list[float]
    per_customer: list[float]
    per_customer_low: list[float]
    per_customer_high: list[float]
    incremental_per_1000: list[float]
    incremental_per_1000_low: list[float]
    incremental_per_1000_high: list[float]


@dataclass(frozen=True)
class PolicyEvaluation:
    n_test: int
    rank_outcome: str
    best_single_arm: int
    selected_learners: dict[str, LearnerName]
    design_shares: list[float]
    bootstrap_resamples: int
    table: list[PolicyResult]
    comparisons: list[Comparison]
    curves: list[PolicyCurve]


def budget_grid(step: float, required: tuple[float, ...]) -> list[float]:
    grid = {round(b, 6) for b in np.arange(0.0, 1.0 + step / 2, step)}
    grid |= {round(b, 6) for b in required}
    return sorted(b for b in grid if 0.0 <= b <= 1.0)


def evaluate_policies(
    test: dict[str, FloatArray],
    selected: dict[str, dict[str, LearnerName]],
    best_single_arm: dict[str, int],
    config: Config,
) -> PolicyEvaluation:
    cfg = config.policy
    alpha = config.experiment.alpha
    rank = cfg.rank_outcome
    logged: IntArray = test["arm"].astype(np.int64)
    n, n_arms = len(logged), len(ARMS)
    design = np.full(n_arms, 1.0 / n_arms)
    outcomes = np.column_stack([test[o] for o in OUTCOMES])
    best_arm = best_single_arm[rank]
    learners = {str(a): selected[rank][str(a)] for a in TREATMENT_ARMS}

    uplift = np.column_stack(
        [test[prediction_key("uplift", rank, a, learners[str(a)])] for a in TREATMENT_ARMS]
    )
    multi_score, multi_arm = best_arm_by_uplift(uplift, TREATMENT_ARMS)
    single_score = uplift[:, TREATMENT_ARMS.index(best_arm)]
    response_score = test[prediction_key("response", rank, best_arm)]
    always_best = np.full(n, best_arm, dtype=np.int64)

    policies: list[Policy] = [email_no_one(n, n_arms)]
    policies += [email_everyone(n, n_arms, a) for a in TREATMENT_ARMS]
    budgets = budget_grid(cfg.grid_step, cfg.budgets)
    for b in budgets:
        policies += [
            email_top_k("uplift_multi", multi_score, multi_arm, b, n_arms),
            email_top_k("uplift_single", single_score, always_best, b, n_arms),
            email_top_k("response", response_score, always_best, b, n_arms),
            email_random(n, n_arms, best_arm, b),
        ]
    weights = np.column_stack([importance_weights(p, logged, design) for p in policies])
    boot = snipw_bootstrap(
        weights,
        outcomes,
        resamples=cfg.bootstrap_resamples,
        seed=derive_seed(config.seed, "policy-bootstrap"),
        labels=[f"{p.name} at budget {p.budget:g}" for p in policies],
    )
    values = [_outcome_values(boot, policies, c, alpha) for c in range(len(policies))]

    index = {(p.name, round(p.budget, 6)): c for c, p in enumerate(policies)}
    fixed = [c for c, p in enumerate(policies) if p.name not in BUDGETED_POLICIES]
    in_table = fixed + [
        index[(name, round(b, 6))] for b in cfg.budgets for name in BUDGETED_POLICIES
    ]
    table = [
        PolicyResult(policies[c].name, policies[c].budget, policies[c].emailed_share, values[c])
        for c in in_table
    ]
    # Random targeting at any budget has the same expected value per email as emailing everyone
    # with that arm, so "vs random" also answers "vs email everyone".
    comparisons = [
        _compare(
            boot,
            policies,
            index[("uplift_multi", round(b, 6))],
            index[(other, round(b, 6))],
            o,
            alpha,
        )
        for b in cfg.budgets
        for other in ("random", "response", "uplift_single")
        for o in range(len(OUTCOMES))
    ]
    curves = [
        _curve(name, o, budgets, [index[(name, round(b, 6))] for b in budgets], values)
        for name in BUDGETED_POLICIES
        for o in range(len(OUTCOMES))
    ]
    return PolicyEvaluation(
        n_test=n,
        rank_outcome=rank,
        best_single_arm=best_arm,
        selected_learners=learners,
        design_shares=design.tolist(),
        bootstrap_resamples=cfg.bootstrap_resamples,
        table=table,
        comparisons=comparisons,
        curves=curves,
    )


def _interval(point: float, draws: FloatArray, alpha: float) -> Interval:
    low, high = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high))


def _incremental(
    boot: BootstrapValues, policies: list[Policy], c: int
) -> tuple[FloatArray, FloatArray]:
    """Point and draws of (V(policy) - V(no one)) * 1000 / emailed share, for every outcome."""
    share = policies[c].emailed_share
    if share == 0:
        nan = np.full(boot.point.shape[1], np.nan)
        return nan, np.full(boot.draws[:, c, :].shape, np.nan)
    # Column 0 is always "no one" (see evaluate_policies).
    scale = 1000.0 / share
    return (boot.point[c] - boot.point[0]) * scale, (boot.draws[:, c] - boot.draws[:, 0]) * scale


def _outcome_values(
    boot: BootstrapValues, policies: list[Policy], c: int, alpha: float
) -> list[OutcomeValue]:
    incr_point, incr_draws = _incremental(boot, policies, c)
    rows: list[OutcomeValue] = []
    for o, outcome in enumerate(OUTCOMES):
        incremental = (
            Interval(float("nan"), float("nan"), float("nan"))
            if np.isnan(incr_point[o])
            else _interval(float(incr_point[o]), incr_draws[:, o], alpha)
        )
        rows.append(
            OutcomeValue(
                outcome=outcome,
                per_customer=_interval(float(boot.point[c, o]), boot.draws[:, c, o], alpha),
                per_customer_ipw=float(boot.point_ipw[c, o]),
                incremental_per_1000_emails=incremental,
            )
        )
    return rows


def _compare(
    boot: BootstrapValues, policies: list[Policy], a: int, b: int, o: int, alpha: float
) -> Comparison:
    point_a, draws_a = _incremental(boot, policies, a)
    point_b, draws_b = _incremental(boot, policies, b)
    return Comparison(
        budget=policies[a].budget,
        outcome=OUTCOMES[o],
        policy_a=policies[a].name,
        policy_b=policies[b].name,
        difference_per_1000_emails=_interval(
            float(point_a[o] - point_b[o]), draws_a[:, o] - draws_b[:, o], alpha
        ),
    )


def _curve(
    name: str, o: int, budgets: list[float], columns: list[int], values: list[list[OutcomeValue]]
) -> PolicyCurve:
    rows = [values[c][o] for c in columns]
    return PolicyCurve(
        policy=name,
        outcome=OUTCOMES[o],
        budgets=budgets,
        per_customer=[r.per_customer.estimate for r in rows],
        per_customer_low=[r.per_customer.low for r in rows],
        per_customer_high=[r.per_customer.high for r in rows],
        incremental_per_1000=[r.incremental_per_1000_emails.estimate for r in rows],
        incremental_per_1000_low=[r.incremental_per_1000_emails.low for r in rows],
        incremental_per_1000_high=[r.incremental_per_1000_emails.high for r in rows],
    )
