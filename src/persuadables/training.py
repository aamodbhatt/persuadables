"""The ``train`` stage: cross-validate every learner on the training split, pick one per
(outcome, arm) by mean Qini coefficient, refit it on the whole training split, and score the
held-out test customers.

Nothing in this stage reads the test outcomes. The test split is touched only to predict, so the
evaluation stage can treat it as a fresh randomized experiment.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from persuadables._types import BoolArray, FloatArray, IntArray
from persuadables.config import BinaryOutcome, Config, LearnerName
from persuadables.data.frame import ARMS, CONTROL_ARM, FEATURES, TREATMENT_ARMS, ExperimentData
from persuadables.io import derive_seed
from persuadables.uplift.explain import UpliftExplanation, explain_uplift
from persuadables.uplift.learners import UpliftLearner, as_explainable, make_learner
from persuadables.uplift.metrics import auuc, qini_coefficient, uplift_at_k
from persuadables.uplift.models import ModelFactory, xgboost_factory

_SHAP_SAMPLE_ROWS = 2000
#: Outcomes counted per split and arm in ``training.json`` (spend is a sum, not a count).
_COUNTED_OUTCOMES: tuple[BinaryOutcome, ...] = ("visit", "conversion")


@dataclass(frozen=True)
class FoldScore:
    fold: int
    qini: float
    auuc: float
    uplift_at: dict[str, float]


@dataclass(frozen=True)
class CrossValidation:
    outcome: BinaryOutcome
    arm: int
    learner: LearnerName
    folds: list[FoldScore]
    seconds: float

    @property
    def mean_qini(self) -> float:
        return statistics.fmean(f.qini for f in self.folds)

    @property
    def sd_qini(self) -> float:
        return statistics.stdev(f.qini for f in self.folds) if len(self.folds) > 1 else 0.0

    @property
    def mean_auuc(self) -> float:
        return statistics.fmean(f.auuc for f in self.folds)


@dataclass(frozen=True)
class TrainingResult:
    cv: list[CrossValidation]
    #: Selected learner per outcome and arm (keys: outcome, then arm as a string).
    selected: dict[str, dict[str, LearnerName]]
    #: The single arm with the larger training-split effect, per outcome.
    best_single_arm: dict[str, int]
    explanations: dict[str, UpliftExplanation]
    n_train: int
    n_test: int
    #: Customers and positive outcomes per split ("train", "test") and arm: how much signal
    #: each model and each test-split estimate actually had.
    outcome_counts: dict[str, dict[str, dict[str, int]]]
    seconds: float
    #: Test-split arrays: ids, arm, outcomes and every model's predictions (see ``prediction_key``).
    test_arrays: dict[str, FloatArray] = field(repr=False)

    def shap_arrays(self) -> dict[str, FloatArray]:
        """Per-row SHAP samples, keyed "<outcome>/<arm>/shap" and "<outcome>/<arm>/features"."""
        arrays: dict[str, FloatArray] = {}
        for key, explanation in self.explanations.items():
            arrays[f"{key}/shap"] = explanation.sample_shap
            arrays[f"{key}/features"] = explanation.sample_features
        return arrays


def prediction_key(kind: str, outcome: str, arm: int, learner: str | None = None) -> str:
    """Name of a prediction array in ``predictions.npz``: kind is "uplift" or "response"."""
    parts = [kind, outcome, str(arm)] + ([learner] if learner else [])
    return "/".join(parts)


def train(
    data: ExperimentData, config: Config, factory: ModelFactory | None = None
) -> TrainingResult:
    started = time.perf_counter()
    cfg = config.uplift
    factory = factory or xgboost_factory(cfg.xgboost)
    is_test = data.is_test
    X = data.matrix(FEATURES)
    X_train, X_test = X[~is_test], X[is_test]
    arm_train = data.arm[~is_test]

    test_arrays: dict[str, FloatArray] = {
        "customer_id": data.numeric("customer_id")[is_test],
        "arm": data.arm[is_test].astype(np.float64),
        "visit": data.outcome("visit")[is_test],
        "conversion": data.outcome("conversion")[is_test],
        "spend": data.outcome("spend")[is_test],
    }
    cv_results: list[CrossValidation] = []
    selected: dict[str, dict[str, LearnerName]] = {}
    best_single_arm: dict[str, int] = {}
    explanations: dict[str, UpliftExplanation] = {}

    for outcome in cfg.outcomes:
        y_train = data.outcome(outcome)[~is_test]
        selected[outcome] = {}
        effects = {a: _train_effect(arm_train, y_train, a) for a in TREATMENT_ARMS}
        best_single_arm[outcome] = max(effects, key=lambda a: effects[a])
        for arm in TREATMENT_ARMS:
            pair = (arm_train == arm) | (arm_train == CONTROL_ARM)
            Xp, yp, tp = X_train[pair], y_train[pair], arm_train[pair] == arm
            fitted: dict[LearnerName, UpliftLearner] = {}
            for name in cfg.learners:
                cv_results.append(_cross_validate(name, Xp, tp, yp, outcome, arm, factory, config))
                learner = make_learner(
                    name,
                    factory,
                    derive_seed(config.seed, "final", outcome, arm, name),
                    dr_crossfit_folds=cfg.dr_crossfit_folds,
                )
                learner.fit(Xp, tp, yp)
                fitted[name] = learner
                test_arrays[prediction_key("uplift", outcome, arm, name)] = learner.predict_uplift(
                    X_test
                )
            scores = {
                c.learner: c.mean_qini for c in cv_results if (c.outcome, c.arm) == (outcome, arm)
            }
            selected[outcome][str(arm)] = max(scores, key=lambda n: scores[n])
            # Explain the best-scoring learner whose uplift has an exact SHAP attribution.
            ranked = sorted(scores, key=lambda n: -scores[n])
            explainable = [e for n in ranked if (e := as_explainable(fitted[n])) is not None]
            if explainable:
                chosen = explainable[0]
                explanations[f"{outcome}/{arm}"] = explain_uplift(
                    chosen,
                    X_test,
                    FEATURES,
                    sample_rows=_SHAP_SAMPLE_ROWS,
                    seed=derive_seed(config.seed, "shap", outcome, arm),
                )
            # Response model for the "target the likely responders" baseline: P(y=1 | x, this
            # arm) for this outcome. The policy baseline uses the one for the ranking outcome.
            response = factory("classification", derive_seed(config.seed, "response", outcome, arm))
            in_arm = arm_train == arm
            response.fit(X_train[in_arm], y_train[in_arm])
            test_arrays[prediction_key("response", outcome, arm)] = response.predict(X_test)

    return TrainingResult(
        cv=cv_results,
        selected=selected,
        best_single_arm=best_single_arm,
        explanations=explanations,
        n_train=int((~is_test).sum()),
        n_test=int(is_test.sum()),
        outcome_counts={
            split: _outcome_counts(data, in_split)
            for split, in_split in (("train", ~is_test), ("test", is_test))
        },
        seconds=time.perf_counter() - started,
        test_arrays=test_arrays,
    )


def _outcome_counts(data: ExperimentData, in_split: BoolArray) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        rows = in_split & (data.arm == arm)
        counts[str(arm)] = {"customers": int(rows.sum())} | {
            outcome: int(data.outcome(outcome)[rows].sum()) for outcome in _COUNTED_OUTCOMES
        }
    return counts


def _train_effect(arm: IntArray, y: FloatArray, treatment: int) -> float:
    return float(y[arm == treatment].mean() - y[arm == CONTROL_ARM].mean())


def _cross_validate(
    name: LearnerName,
    X: FloatArray,
    treated: BoolArray,
    y: FloatArray,
    outcome: BinaryOutcome,
    arm: int,
    factory: ModelFactory,
    config: Config,
) -> CrossValidation:
    started = time.perf_counter()
    cfg = config.uplift
    splitter = StratifiedKFold(
        cfg.cv_folds, shuffle=True, random_state=derive_seed(config.seed, "cv", outcome, arm)
    )
    strata = treated.astype(np.int64) * 2 + y.astype(np.int64)
    folds: list[FoldScore] = []
    for k, (fit_idx, val_idx) in enumerate(splitter.split(X, strata)):
        learner = make_learner(
            name,
            factory,
            derive_seed(config.seed, "cv-model", outcome, arm, name, k),
            dr_crossfit_folds=cfg.dr_crossfit_folds,
        )
        learner.fit(X[fit_idx], treated[fit_idx], y[fit_idx])
        score = learner.predict_uplift(X[val_idx])
        t_val, y_val = treated[val_idx], y[val_idx]
        folds.append(
            FoldScore(
                fold=k,
                qini=qini_coefficient(score, t_val, y_val),
                auuc=auuc(score, t_val, y_val),
                uplift_at={f"{q:g}": uplift_at_k(score, t_val, y_val, q) for q in cfg.uplift_at},
            )
        )
    return CrossValidation(outcome, arm, name, folds, time.perf_counter() - started)


def save_predictions(path: Path, arrays: dict[str, FloatArray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy's stub types **kwds against its own keyword parameters; every value here is an array.
    savez: Callable[..., None] = np.savez_compressed
    savez(path, **arrays)


def load_predictions(path: Path) -> dict[str, FloatArray]:
    if not path.is_file():
        msg = f"{path} not found; run `persuadables train` first"
        raise FileNotFoundError(msg)
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}


def training_summary(result: TrainingResult) -> dict[str, Any]:
    """The JSON-sized, deterministic part of a training run: rerunning the same config gives a
    byte-identical file (wall-clock timings and per-row SHAP samples are stored elsewhere)."""
    return {
        "n_train": result.n_train,
        "n_test": result.n_test,
        "outcome_counts": result.outcome_counts,
        "selected": result.selected,
        "best_single_arm": result.best_single_arm,
        "cross_validation": [
            {
                "outcome": cv.outcome,
                "arm": cv.arm,
                "learner": cv.learner,
                "mean_qini": cv.mean_qini,
                "sd_qini": cv.sd_qini,
                "mean_auuc": cv.mean_auuc,
                "folds": cv.folds,
            }
            for cv in result.cv
        ],
        "shap": {
            key: {
                "learner": e.learner,
                "feature_names": e.feature_names,
                "mean_abs_shap": e.mean_abs_shap,
                "mean_shap": e.mean_shap,
                "base_value": e.base_value,
            }
            for key, e in result.explanations.items()
        },
    }


def training_timings(result: TrainingResult) -> dict[str, Any]:
    return {
        "total_seconds": result.seconds,
        "cross_validation_seconds": {
            f"{cv.outcome}/{cv.arm}/{cv.learner}": cv.seconds for cv in result.cv
        },
    }
