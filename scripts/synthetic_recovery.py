"""Measure how well each learner recovers a KNOWN heterogeneous effect (synthetic randomized
experiment, see ``persuadables.uplift.synthetic``) and write ``results/synthetic_recovery.json``.

    uv run python scripts/synthetic_recovery.py

The test suite asserts lower bounds on the same quantities; this script records the values.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

from persuadables.config import LearnerName, XGBoostConfig
from persuadables.io import write_json
from persuadables.uplift.learners import LEARNER_LABELS, make_learner
from persuadables.uplift.metrics import qini_coefficient
from persuadables.uplift.models import XGBOutcomeModel, xgboost_factory
from persuadables.uplift.synthetic import make_synthetic_experiment

ROOT = Path(__file__).resolve().parents[1]
PARAMS = XGBoostConfig(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=20,
    reg_lambda=5.0,
    n_jobs=4,
)
LEARNERS: tuple[LearnerName, ...] = ("s", "t", "x", "dr", "ct")


def main() -> None:
    train = make_synthetic_experiment(16_000, seed=1)
    test = make_synthetic_experiment(8_000, seed=2)
    factory = xgboost_factory(PARAMS)
    rows = []
    for name in LEARNERS:
        learner = make_learner(name, factory, seed=3)
        learner.fit(train.X, train.treated, train.y)
        predicted = learner.predict_uplift(test.X)
        rows.append(
            {
                "learner": LEARNER_LABELS[name],
                "spearman_with_true_uplift": float(stats.spearmanr(predicted, test.tau).statistic),
                "pehe": math.sqrt(float(np.mean((predicted - test.tau) ** 2))),
                "ate_error": float(predicted.mean() - test.tau.mean()),
                "qini": qini_coefficient(predicted, test.treated, test.y),
            }
        )
    response = XGBOutcomeModel("classification", PARAMS, seed=4)
    response.fit(train.X[train.treated], train.y[train.treated])
    summary = {
        "n_train": len(train.y),
        "n_test": len(test.y),
        "true_ate": float(test.tau.mean()),
        "true_uplift_sd": float(test.tau.std()),
        # Ranking by the true uplift: a reference, not an upper bound on realised outcomes.
        "true_uplift_ranking_qini": qini_coefficient(test.tau, test.treated, test.y),
        "response_model_qini": qini_coefficient(response.predict(test.X), test.treated, test.y),
        "constant_prediction_pehe": float(test.tau.std()),
        "learners": rows,
    }
    write_json(ROOT / "results" / "synthetic_recovery.json", summary)
    for row in rows:
        print(
            f"{row['learner']:<22} rho={row['spearman_with_true_uplift']:.3f} "
            f"PEHE={row['pehe']:.4f} Qini={row['qini']:.3f}"
        )
    reference, response_qini = summary["true_uplift_ranking_qini"], summary["response_model_qini"]
    print(f"true-uplift ranking Qini={reference:.3f}  response model Qini={response_qini:.3f}")


if __name__ == "__main__":
    main()
