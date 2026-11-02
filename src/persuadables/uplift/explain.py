"""SHAP explanations of predicted uplift: which pre-period features make a customer persuadable.

The X- and DR-learners end in regression trees that predict the uplift directly, so exact TreeSHAP
(XGBoost's built-in ``pred_contribs``) attributes the uplift itself, not a response probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persuadables._types import FloatArray
from persuadables.uplift.learners import ExplainableUplift


@dataclass(frozen=True)
class UpliftExplanation:
    learner: str
    feature_names: tuple[str, ...]
    #: Mean |SHAP| per feature over the explained rows (global importance, in uplift units).
    mean_abs_shap: tuple[float, ...]
    #: Mean SHAP per feature: the signed average push on uplift.
    mean_shap: tuple[float, ...]
    base_value: float
    #: A seeded subsample of per-row SHAP values and feature values, for the beeswarm figure.
    sample_shap: FloatArray
    sample_features: FloatArray


def explain_uplift(
    learner: ExplainableUplift,
    X: FloatArray,
    feature_names: tuple[str, ...],
    *,
    sample_rows: int,
    seed: int,
) -> UpliftExplanation:
    if X.shape[1] != len(feature_names):
        msg = f"X has {X.shape[1]} columns but {len(feature_names)} feature names were given"
        raise ValueError(msg)
    contributions = learner.uplift_shap_values(X)
    values, bias = contributions[:, :-1], contributions[:, -1]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(sample_rows, len(X)), replace=False)
    return UpliftExplanation(
        learner=learner.name,
        feature_names=feature_names,
        mean_abs_shap=tuple(float(v) for v in np.abs(values).mean(axis=0)),
        mean_shap=tuple(float(v) for v in values.mean(axis=0)),
        base_value=float(bias.mean()),
        sample_shap=values[sample],
        sample_features=X[sample],
    )
