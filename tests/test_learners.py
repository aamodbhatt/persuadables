"""Every learner must recover a KNOWN heterogeneous effect on synthetic randomized data, and beat
a response model that ranks by baseline propensity instead of persuadability."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from persuadables.config import LearnerName, XGBoostConfig
from persuadables.uplift.explain import explain_uplift
from persuadables.uplift.learners import UpliftLearner, as_explainable, make_learner
from persuadables.uplift.metrics import qini_coefficient
from persuadables.uplift.models import XGBOutcomeModel, xgboost_factory
from persuadables.uplift.synthetic import (
    FEATURE_NAMES,
    SyntheticExperiment,
    make_synthetic_experiment,
)

PARAMS = XGBoostConfig(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=20,
    reg_lambda=5.0,
    n_jobs=2,
)
FACTORY = xgboost_factory(PARAMS)
LEARNERS: tuple[LearnerName, ...] = ("s", "t", "x", "dr", "ct")

Split = tuple[SyntheticExperiment, SyntheticExperiment]


@pytest.fixture(scope="module")
def experiment() -> Split:
    return make_synthetic_experiment(16_000, seed=1), make_synthetic_experiment(8_000, seed=2)


@pytest.fixture(scope="module")
def fitted(experiment: Split) -> dict[LearnerName, UpliftLearner]:
    train, _ = experiment
    models: dict[LearnerName, UpliftLearner] = {}
    for name in LEARNERS:
        learner = make_learner(name, FACTORY, seed=3)
        learner.fit(train.X, train.treated, train.y)
        models[name] = learner
    return models


@pytest.mark.parametrize("name", LEARNERS)
def test_learner_recovers_the_true_effect_ranking(
    name: LearnerName, experiment: Split, fitted: dict[LearnerName, UpliftLearner]
) -> None:
    _, test = experiment
    predicted = fitted[name].predict_uplift(test.X)
    rho = float(stats.spearmanr(predicted, test.tau).statistic)
    true_ranking = qini_coefficient(test.tau, test.treated, test.y)
    model = qini_coefficient(predicted, test.treated, test.y)
    assert rho > 0.8, f"{name}: Spearman correlation with the true uplift is only {rho:.2f}"
    assert model > 0.75 * true_ranking, (
        f"{name}: Qini {model:.3f} vs true-uplift ranking {true_ranking:.3f}"
    )


@pytest.mark.parametrize("name", ["t", "x", "dr"])
def test_learner_estimates_the_average_effect(
    name: LearnerName, experiment: Split, fitted: dict[LearnerName, UpliftLearner]
) -> None:
    _, test = experiment
    predicted = fitted[name].predict_uplift(test.X)
    assert predicted.mean() == pytest.approx(test.tau.mean(), abs=0.02)


def test_uplift_beats_response_targeting(
    experiment: Split, fitted: dict[LearnerName, UpliftLearner]
) -> None:
    train, test = experiment
    response = XGBOutcomeModel("classification", PARAMS, seed=4)
    response.fit(train.X[train.treated], train.y[train.treated])
    by_response = qini_coefficient(response.predict(test.X), test.treated, test.y)
    by_uplift = qini_coefficient(fitted["x"].predict_uplift(test.X), test.treated, test.y)
    assert by_uplift > by_response + 0.04


@pytest.mark.parametrize("name", ["x", "dr"])
def test_shap_attributes_uplift_exactly_and_to_the_right_feature(
    name: LearnerName, experiment: Split, fitted: dict[LearnerName, UpliftLearner]
) -> None:
    _, test = experiment
    learner = as_explainable(fitted[name])
    assert learner is not None
    contributions = learner.uplift_shap_values(test.X)
    np.testing.assert_allclose(contributions.sum(axis=1), learner.predict_uplift(test.X), atol=1e-4)
    explanation = explain_uplift(learner, test.X, FEATURE_NAMES, sample_rows=500, seed=0)
    assert explanation.feature_names[int(np.argmax(explanation.mean_abs_shap))] == "x2"
    assert explanation.sample_shap.shape == (500, len(FEATURE_NAMES))


def test_non_explainable_learners_are_reported_as_such(
    fitted: dict[LearnerName, UpliftLearner],
) -> None:
    assert as_explainable(fitted["s"]) is None
    assert as_explainable(fitted["ct"]) is None


def test_learners_reject_non_binary_outcomes() -> None:
    data = make_synthetic_experiment(200, seed=5)
    with pytest.raises(ValueError, match="binary"):
        make_learner("t", FACTORY, seed=0).fit(data.X, data.treated, data.y * 2.5)


def test_single_class_training_data_gives_a_constant_model() -> None:
    model = XGBOutcomeModel("classification", PARAMS, seed=0)
    X = np.random.default_rng(0).normal(size=(50, 3))
    model.fit(X, np.zeros(50))
    np.testing.assert_array_equal(model.predict(X), np.zeros(50))
    assert model.shap_values(X).shape == (50, 4)
