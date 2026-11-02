"""Meta-learners for the conditional average treatment effect (CATE, "uplift") of one treatment
against control on a binary outcome:

    tau(x) = P(y = 1 | x, treated) - P(y = 1 | x, control)

All learners assume a randomized experiment: the treatment probability ``e`` is a constant set by
the design (estimated as the treated share of the training data), not a function of x.

- S-learner: one model with the treatment flag as a feature; tau = m(x, 1) - m(x, 0).
- T-learner: separate models per arm; tau = m1(x) - m0(x).
- X-learner (Kunzel et al., 2019): impute individual effects with the other arm's model, regress
  them on x, and blend the two effect models by the propensity.
- DR-learner (Kennedy, 2023): regress the cross-fitted doubly robust (AIPW) pseudo-outcome on x.
- Class transformation (Jaskowski and Jaroszewicz, 2012): one classifier on
  z = y*t + (1-y)*(1-t) with the arms reweighted to equal mass; tau = 2 P(z = 1 | x) - 1.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.model_selection import StratifiedKFold

from persuadables._types import BoolArray, FloatArray
from persuadables.config import LearnerName
from persuadables.uplift.models import ModelFactory


class UpliftLearner(Protocol):
    name: LearnerName

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None: ...

    def predict_uplift(self, X: FloatArray) -> FloatArray: ...


class ExplainableUplift(UpliftLearner, Protocol):
    def uplift_shap_values(self, X: FloatArray) -> FloatArray:
        """TreeSHAP attribution of the predicted uplift itself, shape (n, p + 1), bias last.

        Rows sum exactly to ``predict_uplift(X)``.
        """
        ...


def _check_inputs(X: FloatArray, treated: BoolArray, y: FloatArray) -> float:
    if not len(X) == len(treated) == len(y):
        msg = f"X, treated and y must have equal length; got {len(X)}, {len(treated)}, {len(y)}"
        raise ValueError(msg)
    if not np.isin(y, (0.0, 1.0)).all():
        msg = "uplift learners here model binary outcomes; y must be 0/1"
        raise ValueError(msg)
    e = float(treated.mean())
    if not 0 < e < 1:
        msg = "both arms need at least one customer to estimate an uplift"
        raise ValueError(msg)
    return e


class SLearner:
    name: LearnerName = "s"

    def __init__(self, factory: ModelFactory, seed: int) -> None:
        self._model = factory("classification", seed)

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None:
        _check_inputs(X, treated, y)
        self._model.fit(np.column_stack([X, treated.astype(np.float64)]), y)

    def predict_uplift(self, X: FloatArray) -> FloatArray:
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return self._model.predict(np.column_stack([X, ones])) - self._model.predict(
            np.column_stack([X, zeros])
        )


class TLearner:
    name: LearnerName = "t"

    def __init__(self, factory: ModelFactory, seed: int) -> None:
        self._treated_model = factory("classification", seed)
        self._control_model = factory("classification", seed + 1)

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None:
        _check_inputs(X, treated, y)
        self._treated_model.fit(X[treated], y[treated])
        self._control_model.fit(X[~treated], y[~treated])

    def predict_uplift(self, X: FloatArray) -> FloatArray:
        return self._treated_model.predict(X) - self._control_model.predict(X)


class XLearner:
    name: LearnerName = "x"

    def __init__(self, factory: ModelFactory, seed: int) -> None:
        self._m1 = factory("classification", seed)
        self._m0 = factory("classification", seed + 1)
        self._tau1 = factory("regression", seed + 2)
        self._tau0 = factory("regression", seed + 3)
        self._e = 0.5

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None:
        self._e = _check_inputs(X, treated, y)
        X1, y1, X0, y0 = X[treated], y[treated], X[~treated], y[~treated]
        self._m1.fit(X1, y1)
        self._m0.fit(X0, y0)
        # Imputed individual effects: observed outcome minus the other arm's prediction.
        self._tau1.fit(X1, y1 - self._m0.predict(X1))
        self._tau0.fit(X0, self._m1.predict(X0) - y0)

    def predict_uplift(self, X: FloatArray) -> FloatArray:
        # Kunzel's blend g(x) = e(x): weight e on tau0, whose imputed effects use m1, the outcome
        # model trained on the treated arm (more accurate when the treated arm is the larger).
        return self._e * self._tau0.predict(X) + (1 - self._e) * self._tau1.predict(X)

    def uplift_shap_values(self, X: FloatArray) -> FloatArray:
        # SHAP values are linear in the model, so the blend's attribution is the blend of theirs.
        return self._e * self._tau0.shap_values(X) + (1 - self._e) * self._tau1.shap_values(X)


class DRLearner:
    name: LearnerName = "dr"

    def __init__(self, factory: ModelFactory, seed: int, crossfit_folds: int = 2) -> None:
        self._factory = factory
        self._seed = seed
        self._folds = crossfit_folds
        self._final = factory("regression", seed + 100)

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None:
        e = _check_inputs(X, treated, y)
        mu1, mu0 = np.empty(len(y)), np.empty(len(y))
        # Cross-fitting: each row's nuisance predictions come from models that never saw it, so
        # their overfitting cannot leak into the pseudo-outcome.
        folds = StratifiedKFold(self._folds, shuffle=True, random_state=self._seed)
        strata = treated.astype(np.int64) * 2 + y.astype(np.int64)
        for k, (fit_idx, out_idx) in enumerate(folds.split(X, strata)):
            t_fit = treated[fit_idx]
            m1 = self._factory("classification", self._seed + 10 * k)
            m0 = self._factory("classification", self._seed + 10 * k + 1)
            m1.fit(X[fit_idx][t_fit], y[fit_idx][t_fit])
            m0.fit(X[fit_idx][~t_fit], y[fit_idx][~t_fit])
            mu1[out_idx] = m1.predict(X[out_idx])
            mu0[out_idx] = m0.predict(X[out_idx])
        t = treated.astype(np.float64)
        pseudo = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self._final.fit(X, pseudo)

    def predict_uplift(self, X: FloatArray) -> FloatArray:
        return self._final.predict(X)

    def uplift_shap_values(self, X: FloatArray) -> FloatArray:
        return self._final.shap_values(X)


class ClassTransformationLearner:
    name: LearnerName = "ct"

    def __init__(self, factory: ModelFactory, seed: int) -> None:
        self._model = factory("classification", seed)

    def fit(self, X: FloatArray, treated: BoolArray, y: FloatArray) -> None:
        e = _check_inputs(X, treated, y)
        z = y * treated + (1 - y) * (~treated)
        # Equal total weight per arm makes the identity tau = 2 P(z=1|x) - 1 hold for any e.
        weights = np.where(treated, 0.5 / e, 0.5 / (1 - e))
        self._model.fit(X, z.astype(np.float64), sample_weight=weights)

    def predict_uplift(self, X: FloatArray) -> FloatArray:
        return 2 * self._model.predict(X) - 1


def make_learner(
    name: LearnerName, factory: ModelFactory, seed: int, *, dr_crossfit_folds: int = 2
) -> UpliftLearner:
    match name:
        case "s":
            return SLearner(factory, seed)
        case "t":
            return TLearner(factory, seed)
        case "x":
            return XLearner(factory, seed)
        case "dr":
            return DRLearner(factory, seed, dr_crossfit_folds)
        case "ct":
            return ClassTransformationLearner(factory, seed)


LEARNER_LABELS: dict[LearnerName, str] = {
    "s": "S-learner",
    "t": "T-learner",
    "x": "X-learner",
    "dr": "DR-learner",
    "ct": "Class transformation",
}


def as_explainable(learner: UpliftLearner) -> ExplainableUplift | None:
    """The learner itself if its uplift has an exact SHAP attribution (X- and DR-learner)."""
    return learner if isinstance(learner, XLearner | DRLearner) else None
