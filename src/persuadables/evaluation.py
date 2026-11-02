"""The ``evaluate`` stage: score every learner on the held-out test split (Qini, AUUC, uplift@k,
calibration by decile), then value the targeting policies built from the selected learners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import BoolArray, FloatArray
from persuadables.config import Config, LearnerName
from persuadables.data.frame import CONTROL_ARM, TREATMENT_ARMS
from persuadables.io import derive_seed
from persuadables.policy.evaluation import Interval, PolicyEvaluation, evaluate_policies
from persuadables.training import prediction_key
from persuadables.uplift.metrics import (
    Calibration,
    Curve,
    auuc,
    calibration_by_bin,
    perfect_qini_curve,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)

#: Points at which stored curves are sampled (share of customers targeted).
_CURVE_POINTS = 101


@dataclass(frozen=True)
class SampledCurve:
    """A Qini curve resampled on an even grid of the share targeted, per customer in the slice."""

    share: list[float]
    incremental_per_customer: list[float]


@dataclass(frozen=True)
class LearnerTestScore:
    outcome: str
    arm: int
    learner: LearnerName
    selected: bool
    qini: Interval
    auuc: float
    uplift_at: dict[str, float]
    calibration: Calibration
    qini_curve: SampledCurve


@dataclass(frozen=True)
class UpliftEvaluation:
    n_test_pairs: dict[str, int]
    scores: list[LearnerTestScore]
    #: Oracle and random reference curves per (outcome, arm), keyed "outcome/arm".
    perfect_curves: dict[str, SampledCurve]
    random_curves: dict[str, SampledCurve]


def sample_curve(curve: Curve, n: int) -> SampledCurve:
    share = np.linspace(0.0, 1.0, _CURVE_POINTS)
    values = np.interp(share * n, curve.x, curve.y) / n
    return SampledCurve(share.round(4).tolist(), values.tolist())


def evaluate_uplift(
    test: dict[str, FloatArray], selected: dict[str, dict[str, LearnerName]], config: Config
) -> UpliftEvaluation:
    cfg = config.uplift
    arm = test["arm"].astype(np.int64)
    scores: list[LearnerTestScore] = []
    perfect: dict[str, SampledCurve] = {}
    random: dict[str, SampledCurve] = {}
    sizes: dict[str, int] = {}
    for outcome in cfg.outcomes:
        for treatment in TREATMENT_ARMS:
            pair = (arm == treatment) | (arm == CONTROL_ARM)
            treated: BoolArray = arm[pair] == treatment
            y = test[outcome][pair]
            n = int(pair.sum())
            key = f"{outcome}/{treatment}"
            sizes[str(treatment)] = n
            perfect[key] = sample_curve(perfect_qini_curve(treated, y), n)
            total = qini_curve(np.zeros(n), treated, y).y[-1]
            random[key] = SampledCurve([0.0, 1.0], [0.0, float(total) / n])
            for learner in cfg.learners:
                score = test[prediction_key("uplift", outcome, treatment, learner)][pair]
                scores.append(
                    LearnerTestScore(
                        outcome=outcome,
                        arm=treatment,
                        learner=learner,
                        selected=selected[outcome][str(treatment)] == learner,
                        qini=_qini_interval(score, treated, y, config, key, learner),
                        auuc=auuc(score, treated, y),
                        uplift_at={
                            f"{k:g}": uplift_at_k(score, treated, y, k) for k in cfg.uplift_at
                        },
                        calibration=calibration_by_bin(
                            score, treated, y, bins=cfg.calibration_bins
                        ),
                        qini_curve=sample_curve(qini_curve(score, treated, y), n),
                    )
                )
    return UpliftEvaluation(sizes, scores, perfect, random)


def _qini_interval(
    score: FloatArray, treated: BoolArray, y: FloatArray, config: Config, key: str, learner: str
) -> Interval:
    """Qini coefficient with a percentile bootstrap interval over test customers."""
    rng = np.random.default_rng(derive_seed(config.seed, "qini-bootstrap", key, learner))
    n = len(score)
    draws = np.empty(config.uplift.bootstrap_resamples)
    for b in range(len(draws)):
        idx = rng.integers(0, n, size=n)
        draws[b] = qini_coefficient(score[idx], treated[idx], y[idx])
    alpha = config.experiment.alpha
    low, high = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(qini_coefficient(score, treated, y), float(low), float(high))


@dataclass(frozen=True)
class Evaluation:
    uplift: UpliftEvaluation
    policy: PolicyEvaluation


def evaluate(
    test: dict[str, FloatArray],
    selected: dict[str, dict[str, LearnerName]],
    best_single_arm: dict[str, int],
    config: Config,
) -> Evaluation:
    return Evaluation(
        uplift=evaluate_uplift(test, selected, config),
        policy=evaluate_policies(test, selected, best_single_arm, config),
    )
