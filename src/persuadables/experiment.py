"""The experiment readout: is the randomization sound, what did each email do to each metric,
how precise are we, and does the effect differ across pre-registered segments?

Everything here uses the full randomized sample (train and test): the readout makes no modelling
choices that could overfit, so there is nothing to hold out from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from persuadables._types import BoolArray, FloatArray
from persuadables.config import OUTCOMES, Config, ExperimentConfig, Outcome
from persuadables.data.frame import (
    ARMS,
    CONTROL_ARM,
    TREATMENT_ARMS,
    ExperimentData,
    segment_levels,
)
from persuadables.io import derive_seed
from persuadables.stats.cuped import CupedResult, cuped_effect
from persuadables.stats.design import (
    BalanceRow,
    SampleRatioResult,
    balance_table,
    sample_ratio_mismatch,
)
from persuadables.stats.heterogeneity import CochranQ, cochran_q
from persuadables.stats.inference import (
    BootstrapInterval,
    EffectEstimate,
    bootstrap_mean_diff,
    two_proportion_ztest,
    welch_ttest,
)
from persuadables.stats.multiplicity import holm_adjust
from persuadables.stats.power import (
    minimum_detectable_effect,
    power_two_sample,
    sample_size_per_arm,
)

BINARY_OUTCOMES: frozenset[Outcome] = frozenset({"visit", "conversion"})


@dataclass(frozen=True)
class PowerSummary:
    """Precision of the realised design for one arm-vs-control comparison."""

    sigma: float
    mde: float
    mde_relative: float
    #: The same MDE with CUPED's variance reduction applied to sigma.
    mde_cuped: float
    #: The configured planning lift, relative to the control mean (e.g. 0.1 for +10%).
    reference_relative_lift: float
    #: Power the realised arms had to detect that lift (a fixed effect, not "observed power").
    power_at_reference_lift: float | None
    #: Customers per arm (two equal arms) needed to detect it at the configured power.
    n_per_arm_for_reference_lift: int | None


@dataclass(frozen=True)
class ArmEffect:
    arm: int
    arm_name: str
    metric: Outcome
    estimate: EffectEstimate
    bootstrap: BootstrapInterval
    cuped: CupedResult
    #: Holm-adjusted p-value across every (arm, metric) pair in the readout.
    p_holm: float
    power: PowerSummary


@dataclass(frozen=True)
class SegmentEffect:
    segment: str
    level: str
    arm: int
    metric: Outcome
    estimate: EffectEstimate


@dataclass(frozen=True)
class SegmentTest:
    """Does the effect of ``arm`` on ``metric`` differ across the levels of ``segment``?"""

    segment: str
    arm: int
    metric: Outcome
    cochran: CochranQ
    #: Holm-adjusted across every (segment, arm, metric) heterogeneity test.
    p_holm: float


@dataclass(frozen=True)
class Readout:
    n_customers: int
    arm_counts: dict[str, int]
    #: Share of customers with zero spend: why spend gets a bootstrap next to Welch's t-test.
    spend_zero_share: float
    srm: SampleRatioResult
    balance: list[BalanceRow]
    effects: list[ArmEffect]
    segment_effects: list[SegmentEffect]
    segment_tests: list[SegmentTest]


def estimate_effect(
    y_treatment: FloatArray, y_control: FloatArray, metric: Outcome, *, alpha: float
) -> EffectEstimate:
    """z-test for the binary metrics, Welch's t-test for spend."""
    if metric in BINARY_OUTCOMES:
        return two_proportion_ztest(
            int(y_treatment.sum()),
            len(y_treatment),
            int(y_control.sum()),
            len(y_control),
            alpha=alpha,
        )
    return welch_ttest(y_treatment, y_control, alpha=alpha)


def run_readout(data: ExperimentData, config: Config) -> Readout:
    cfg = config.experiment
    arm = data.arm
    counts = tuple(int((arm == a).sum()) for a in ARMS)
    srm = sample_ratio_mismatch(counts, tuple(1 / len(ARMS) for _ in ARMS))
    balance = balance_table(
        arm,
        data.matrix(cfg.balance_covariates),
        cfg.balance_covariates,
        control=CONTROL_ARM,
        treatments=TREATMENT_ARMS,
    )
    segment_effects = _segment_effects(data, config)
    return Readout(
        n_customers=len(data),
        arm_counts={ARMS[a]: c for a, c in zip(ARMS, counts, strict=True)},
        spend_zero_share=float((data.outcome("spend") == 0).mean()),
        srm=srm,
        balance=balance,
        effects=_arm_effects(data, config),
        segment_effects=segment_effects,
        segment_tests=_segment_tests(segment_effects, config),
    )


def _arm_effects(data: ExperimentData, config: Config) -> list[ArmEffect]:
    cfg = config.experiment
    covariates = data.matrix(cfg.cuped_covariates)
    pending: list[tuple[int, Outcome, EffectEstimate, BootstrapInterval, CupedResult, PowerSummary]]
    pending = []
    for treatment in TREATMENT_ARMS:
        pair = (data.arm == treatment) | (data.arm == CONTROL_ARM)
        treated: BoolArray = data.arm[pair] == treatment
        for metric in OUTCOMES:
            y = data.outcome(metric)[pair]
            estimate = estimate_effect(y[treated], y[~treated], metric, alpha=cfg.alpha)
            boot = bootstrap_mean_diff(
                y[treated],
                y[~treated],
                resamples=cfg.bootstrap_resamples,
                alpha=cfg.alpha,
                seed=derive_seed(config.seed, "bootstrap", treatment, metric),
            )
            cuped = cuped_effect(y, covariates[pair], treated, alpha=cfg.alpha)
            power = _power_summary(
                y[~treated],
                metric,
                int(treated.sum()),
                estimate.control_mean,
                cuped.variance_reduction,
                cfg,
            )
            pending.append((treatment, metric, estimate, boot, cuped, power))
    p_holm = holm_adjust([row[2].p_value for row in pending])
    return [
        ArmEffect(treatment, ARMS[treatment], metric, estimate, boot, cuped, float(p), power)
        for (treatment, metric, estimate, boot, cuped, power), p in zip(
            pending, p_holm, strict=True
        )
    ]


def _power_summary(
    y_control: FloatArray,
    metric: Outcome,
    n_treatment: int,
    control_mean: float,
    variance_reduction: float,
    cfg: ExperimentConfig,
) -> PowerSummary:
    sigma = _control_sigma(y_control, metric)
    n_control = len(y_control)
    mde = minimum_detectable_effect(sigma, n_treatment, n_control, alpha=cfg.alpha, power=cfg.power)
    lift = cfg.reference_relative_lift
    # A zero control mean (or zero variance) has no meaningful relative lift to plan for.
    plannable = control_mean != 0 and sigma > 0
    reference_effect = lift * control_mean
    return PowerSummary(
        sigma=sigma,
        mde=mde,
        mde_relative=mde / control_mean if control_mean else math.nan,
        mde_cuped=mde * math.sqrt(1 - variance_reduction),
        reference_relative_lift=lift,
        power_at_reference_lift=(
            power_two_sample(reference_effect, sigma, n_treatment, n_control, alpha=cfg.alpha)
            if plannable
            else None
        ),
        n_per_arm_for_reference_lift=(
            sample_size_per_arm(reference_effect, sigma, alpha=cfg.alpha, power=cfg.power)
            if plannable
            else None
        ),
    )


def _control_sigma(y_control: FloatArray, metric: Outcome) -> float:
    if metric in BINARY_OUTCOMES:
        rate = float(y_control.mean())
        return math.sqrt(rate * (1 - rate))
    return float(np.std(y_control, ddof=1))


def _segment_effects(data: ExperimentData, config: Config) -> list[SegmentEffect]:
    effects: list[SegmentEffect] = []
    for segment in config.experiment.segments:
        values = data.categorical(segment)
        for level in segment_levels(values, segment):
            in_level = values == level
            for treatment in TREATMENT_ARMS:
                is_t = in_level & (data.arm == treatment)
                is_c = in_level & (data.arm == CONTROL_ARM)
                if is_t.sum() < 2 or is_c.sum() < 2:
                    continue
                for metric in OUTCOMES:
                    y = data.outcome(metric)
                    estimate = estimate_effect(
                        y[is_t], y[is_c], metric, alpha=config.experiment.alpha
                    )
                    effects.append(SegmentEffect(segment, level, treatment, metric, estimate))
    return effects


def _segment_tests(effects: list[SegmentEffect], config: Config) -> list[SegmentTest]:
    raw: list[tuple[str, int, Outcome, CochranQ]] = []
    for segment in config.experiment.segments:
        for treatment in TREATMENT_ARMS:
            for metric in OUTCOMES:
                rows = [
                    e
                    for e in effects
                    if e.segment == segment and e.arm == treatment and e.metric == metric
                ]
                result = cochran_q(
                    np.array([r.estimate.diff for r in rows]),
                    np.array([r.estimate.se for r in rows]),
                )
                raw.append((segment, treatment, metric, result))
    adjusted = holm_adjust([r[3].p_value for r in raw])
    return [
        SegmentTest(segment, treatment, metric, result, float(p))
        for (segment, treatment, metric, result), p in zip(raw, adjusted, strict=True)
    ]
