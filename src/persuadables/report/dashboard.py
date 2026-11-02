"""The self-contained HTML dashboard: experiment readout, segment effects, Qini curves,
calibration, SHAP and an interactive policy simulator.

It is a static file (no server): the results are embedded as JSON and drawn by Plotly in the
browser, so it can be opened from disk, attached to an email or hosted on any static host.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import plotly
from plotly.offline import get_plotlyjs

from persuadables.config import Config
from persuadables.data.frame import ARMS, FEATURES, TREATMENT_ARMS
from persuadables.io import read_json, to_jsonable
from persuadables.report.figures import PER_1000_LABELS, SEGMENT_TITLES
from persuadables.report.style import FEATURE_LABELS, METRIC_LABELS, policy_labels
from persuadables.uplift.learners import LEARNER_LABELS


def build_payload(config: Config) -> dict[str, Any]:
    results = config.output.results_dir
    readout = read_json(results / "readout.json")
    training = read_json(results / "training.json")
    uplift = read_json(results / "uplift_test.json")
    policy = read_json(results / "policy.json")
    validation = read_json(results / "data_validation.json")

    effects = [
        {
            "arm": e["arm"],
            "armName": e["arm_name"],
            "metric": e["metric"],
            "treatment": e["estimate"]["treatment_mean"],
            "control": e["estimate"]["control_mean"],
            "diff": e["estimate"]["diff"],
            "low": e["estimate"]["ci_low"],
            "high": e["estimate"]["ci_high"],
            "bootLow": e["bootstrap"]["low"],
            "bootHigh": e["bootstrap"]["high"],
            "cupedLow": e["cuped"]["adjusted"]["ci_low"],
            "cupedHigh": e["cuped"]["adjusted"]["ci_high"],
            "varianceReduction": e["cuped"]["variance_reduction"],
            "pHolm": e["p_holm"],
            "mde": e["power"]["mde"],
            "mdeCuped": e["power"]["mde_cuped"],
            "method": e["estimate"]["method"],
        }
        for e in readout["effects"]
    ]
    segments = [
        {
            "segment": s["segment"],
            "level": s["level"],
            "arm": s["arm"],
            "metric": s["metric"],
            "diff": s["estimate"]["diff"],
            "low": s["estimate"]["ci_low"],
            "high": s["estimate"]["ci_high"],
        }
        for s in readout["segment_effects"]
    ]
    segment_tests = [
        {"segment": t["segment"], "arm": t["arm"], "metric": t["metric"], "pHolm": t["p_holm"]}
        for t in readout["segment_tests"]
    ]
    qini = [
        {
            "outcome": s["outcome"],
            "arm": s["arm"],
            "learner": s["learner"],
            "selected": s["selected"],
            "qini": s["qini"],
            "auuc": s["auuc"],
            "upliftAt": s["uplift_at"],
            "share": s["qini_curve"]["share"],
            "curve": s["qini_curve"]["incremental_per_customer"],
            "calibration": s["calibration"],
        }
        for s in uplift["scores"]
    ]
    shap = {
        key: {
            "learner": value["learner"],
            "features": [FEATURE_LABELS[f] for f in value["feature_names"]],
            "meanAbs": value["mean_abs_shap"],
            "mean": value["mean_shap"],
        }
        for key, value in training["shap"].items()
    }
    return {
        "meta": {
            "customers": readout["n_customers"],
            "armCounts": readout["arm_counts"],
            "nTrain": training["n_train"],
            "nTest": training["n_test"],
            "srmP": readout["srm"]["p_value"],
            "maxSmd": max(abs(b["smd"]) for b in readout["balance"]),
            "checksPassed": sum(c["passed"] for c in validation["checks"]),
            "checks": len(validation["checks"]),
            "rankOutcome": policy["rank_outcome"],
            "headlineBudget": config.policy.headline_budget,
            "bestArm": policy["best_single_arm"],
            "selected": training["selected"],
            "policyResamples": policy["bootstrap_resamples"],
            "nLearners": len(config.uplift.learners),
            "cvFolds": config.uplift.cv_folds,
            "seed": config.seed,
            "alpha": config.experiment.alpha,
        },
        "labels": {
            "arms": {str(k): v for k, v in ARMS.items()},
            "treatmentArms": list(TREATMENT_ARMS),
            "metrics": METRIC_LABELS,
            "per1000": PER_1000_LABELS,
            "learners": LEARNER_LABELS,
            "policies": policy_labels(policy["rank_outcome"], ARMS[policy["best_single_arm"]]),
            "segments": SEGMENT_TITLES,
            "features": [FEATURE_LABELS[f] for f in FEATURES],
        },
        "effects": effects,
        "segments": segments,
        "segmentTests": segment_tests,
        "qini": qini,
        "randomCurves": uplift["random_curves"],
        "shap": shap,
        "policy": {
            "table": policy["table"],
            "curves": policy["curves"],
            "comparisons": policy["comparisons"],
        },
    }


def write_dashboard(config: Config) -> Path:
    payload = json.dumps(to_jsonable(build_payload(config)), separators=(",", ":"), allow_nan=False)
    template = resources.files("persuadables.report").joinpath("template.html").read_text("utf-8")
    if config.output.plotly_js == "inline":
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        version = plotly.offline.get_plotlyjs_version()
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'
    # The payload is JSON inside a <script> block: escape "</" so no value can close the tag.
    html = template.replace("<!--PLOTLY-->", plotly_tag).replace(
        "/*PAYLOAD*/null", payload.replace("</", "<\\/")
    )
    path = config.output.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", newline="\n")
    return path
