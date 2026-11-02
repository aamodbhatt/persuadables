"""Base models behind the meta-learners: one small interface, XGBoost underneath.

Meta-learners only need "fit E[y | x]" and "predict it". Keeping that behind ``OutcomeModel``
means the learners are tested against the same interface they run on, and the XGBoost settings
live in exactly one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np
import xgboost as xgb

from persuadables._types import FloatArray
from persuadables.config import XGBoostConfig

Task = Literal["classification", "regression"]


class OutcomeModel(Protocol):
    def fit(self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None) -> None:
        """Fit E[y | x] (a probability for classification)."""

    def predict(self, X: FloatArray) -> FloatArray:
        """E[y | x] for each row of X."""

    def shap_values(self, X: FloatArray) -> FloatArray:
        """Exact TreeSHAP contributions, shape (n, p + 1); the last column is the bias term.

        In output space for regression and log-odds space for classification; rows sum to the raw
        model output.
        """


#: Builds a fresh, unfitted model for a task, seeded for reproducibility.
ModelFactory = Callable[[Task, int], OutcomeModel]


class XGBOutcomeModel:
    """An XGBoost classifier or regressor behind the ``OutcomeModel`` interface.

    A training set with a single class (possible for a rare outcome in a small fold) gets a
    constant model instead of an XGBoost error: the best estimate of P(y=1) is then that class.
    """

    def __init__(self, task: Task, params: XGBoostConfig, seed: int) -> None:
        self._task = task
        self._params = params
        self._seed = seed
        self._model: xgb.XGBClassifier | xgb.XGBRegressor | None = None
        self._constant: float | None = None

    def fit(self, X: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None) -> None:
        if len(X) != len(y) or len(y) == 0:
            msg = f"need matching, non-empty X and y; got {len(X)} and {len(y)} rows"
            raise ValueError(msg)
        common = {
            "n_estimators": self._params.n_estimators,
            "max_depth": self._params.max_depth,
            "learning_rate": self._params.learning_rate,
            "subsample": self._params.subsample,
            "colsample_bytree": self._params.colsample_bytree,
            "min_child_weight": self._params.min_child_weight,
            "reg_lambda": self._params.reg_lambda,
            "n_jobs": self._params.n_jobs,
            "random_state": self._seed,
            "tree_method": "hist",
        }
        if self._task == "classification":
            labels = np.unique(y)
            if len(labels) == 1:
                self._constant, self._model = float(labels[0]), None
                return
            self._model = xgb.XGBClassifier(objective="binary:logistic", **common)
        else:
            self._model = xgb.XGBRegressor(objective="reg:squarederror", **common)
        self._constant = None
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X: FloatArray) -> FloatArray:
        if self._constant is not None:
            return np.full(len(X), self._constant)
        model = self._fitted()
        if isinstance(model, xgb.XGBClassifier):
            return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
        return np.asarray(model.predict(X), dtype=np.float64)

    def shap_values(self, X: FloatArray) -> FloatArray:
        if self._constant is not None:
            values = np.zeros((len(X), X.shape[1] + 1))
            values[:, -1] = self._constant
            return values
        contributions = self._fitted().get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return np.asarray(contributions, dtype=np.float64)

    def _fitted(self) -> xgb.XGBClassifier | xgb.XGBRegressor:
        if self._model is None:
            msg = "model is not fitted; call fit() first"
            raise RuntimeError(msg)
        return self._model


def xgboost_factory(params: XGBoostConfig) -> ModelFactory:
    def make(task: Task, seed: int) -> OutcomeModel:
        return XGBOutcomeModel(task, params, seed)

    return make
