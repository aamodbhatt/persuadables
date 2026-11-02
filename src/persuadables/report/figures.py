"""Static result figures for the README, drawn from the JSON files in ``results/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from persuadables.config import OUTCOMES, Config
from persuadables.data.frame import ARMS, FEATURES, SEGMENT_LEVELS, TREATMENT_ARMS
from persuadables.io import read_json
from persuadables.report.style import (
    ARM_COLORS,
    BASELINE,
    FEATURE_LABELS,
    GRID,
    INK_SECONDARY,
    METRIC_LABELS,
    MUTED,
    POLICY_COLORS,
    SERIES,
    new_figure,
    percent,
    policy_labels,
    save,
)
from persuadables.training import load_predictions
from persuadables.uplift.learners import LEARNER_LABELS

SEGMENT_TITLES = {
    "purchase_category": "Past purchases",
    "recency_band": "Recency",
    "history_tier": "Past-year spend",
    "channel": "Channel",
    "zip_area": "Area",
    "tenure": "Tenure",
}
PER_1000_LABELS = {
    "visit": "Incremental visits",
    "conversion": "Incremental conversions",
    "spend": "Incremental revenue ($)",
}


def _results(config: Config, name: str) -> Any:
    return read_json(config.output.results_dir / name)


def write_result_figures(config: Config) -> list[Path]:
    out = config.output.figures_dir
    readout = _results(config, "readout.json")
    uplift = _results(config, "uplift_test.json")
    training = _results(config, "training.json")
    policy = _results(config, "policy.json")
    shap = load_predictions(config.output.artifacts_dir / "shap_sample.npz")
    return [
        _effects(readout, config.experiment.alpha, out / "effects.png"),
        _segments(readout, out / "segments.png"),
        _qini(uplift, config, out / "qini.png"),
        _learners(uplift, training, config, out / "learners.png"),
        _calibration(uplift, config, out / "calibration.png"),
        _shap(training, shap, config, out / "shap.png"),
        _policy(policy, config, out / "policy.png"),
    ]


def _effects(readout: Any, alpha: float, path: Path) -> Path:
    fig, axes = new_figure(8.5, 3.6)
    ax = axes[0]
    rows = readout["effects"]
    labels = []
    for i, row in enumerate(reversed(rows)):
        base = row["estimate"]["control_mean"]
        raw, adj = row["estimate"], row["cuped"]["adjusted"]
        y = i
        ax.plot(
            [raw["ci_low"] / base, raw["ci_high"] / base],
            [y + 0.12] * 2,
            color=BASELINE,
            lw=4,
            solid_capstyle="round",
        )
        ax.plot(
            [adj["ci_low"] / base, adj["ci_high"] / base],
            [y - 0.12] * 2,
            color=ARM_COLORS[row["arm"]],
            lw=2.5,
            solid_capstyle="round",
        )
        ax.scatter(
            [raw["diff"] / base],
            [y],
            s=40,
            color=ARM_COLORS[row["arm"]],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{raw['diff'] / base:+.0%}   Holm p = {row['p_holm']:.0e}",
            (raw["ci_high"] / base, y),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
        labels.append(f"{row['arm_name']}: {METRIC_LABELS[row['metric']].lower()}")
    ax.set_yticks(range(len(rows)), labels)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.xaxis.set_major_formatter(percent())
    ax.set_xlim(
        left=-0.05,
        right=max(r["estimate"]["ci_high"] / r["estimate"]["control_mean"] for r in rows) * 1.45,
    )
    ax.set_xlabel("Lift relative to no email (95% CI: grey = raw, colour = CUPED)")
    significant = sum(r["p_holm"] < alpha for r in rows)
    ax.set_title(
        f"Treatment effects: {significant} of {len(rows)} significant after Holm (alpha {alpha})"
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, path)


def _segments(readout: Any, path: Path) -> Path:
    tests = {(t["segment"], t["arm"], t["metric"]): t for t in readout["segment_tests"]}
    fig, axes = new_figure(11.5, 6.2, 2, 3)
    for ax, segment in zip(axes, SEGMENT_TITLES, strict=True):
        rows = [
            e
            for e in readout["segment_effects"]
            if e["segment"] == segment and e["metric"] == "visit"
        ]
        levels = [lv for lv in SEGMENT_LEVELS[segment] if any(r["level"] == lv for r in rows)]
        for offset, arm in zip((0.14, -0.14), TREATMENT_ARMS, strict=True):
            arm_rows = {r["level"]: r["estimate"] for r in rows if r["arm"] == arm}
            ys = [len(levels) - 1 - i + offset for i in range(len(levels))]
            est = [arm_rows[lv] for lv in levels]
            ax.hlines(
                ys,
                [e["ci_low"] for e in est],
                [e["ci_high"] for e in est],
                color=ARM_COLORS[arm],
                lw=2,
            )
            ax.scatter(
                [e["diff"] for e in est],
                ys,
                s=30,
                color=ARM_COLORS[arm],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
                label=ARMS[arm],
            )
        ax.set_yticks(range(len(levels)), list(reversed(levels)))
        p = [tests[(segment, a, "visit")]["p_holm"] for a in TREATMENT_ARMS]
        ax.set_title(
            f"{SEGMENT_TITLES[segment]}  (Holm p: M {p[0]:.1g}, W {p[1]:.1g})", fontsize=10
        )
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.xaxis.set_major_formatter(percent())
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(
        "Effect on visit rate by pre-registered segment (points: difference, lines: 95% CI); "
        "p tests whether the effect differs across levels",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _panel_key(outcome: str, arm: int) -> str:
    return f"{outcome}/{arm}"


def _qini(uplift: Any, config: Config, path: Path) -> Path:
    outcomes = config.uplift.outcomes
    fig, axes = new_figure(10.5, 3.2 * len(outcomes), len(outcomes), 2)
    for row, outcome in enumerate(outcomes):
        for col, arm in enumerate(TREATMENT_ARMS):
            ax = axes[row * 2 + col]
            key = _panel_key(outcome, arm)
            random = uplift["random_curves"][key]
            scores = [s for s in uplift["scores"] if s["outcome"] == outcome and s["arm"] == arm]
            for s in sorted(scores, key=lambda s: s["selected"]):
                curve = s["qini_curve"]
                name = LEARNER_LABELS[s["learner"]]
                ax.plot(
                    curve["share"],
                    curve["incremental_per_customer"],
                    color=SERIES[0] if s["selected"] else BASELINE,
                    lw=2.2 if s["selected"] else 1.2,
                    label=(
                        f"{name} (selected), Qini {s['qini']['estimate']:.3f}"
                        if s["selected"]
                        else None
                    ),
                    zorder=3 if s["selected"] else 2,
                )
            ax.plot(
                random["share"],
                random["incremental_per_customer"],
                color=MUTED,
                lw=1.2,
                label="Random targeting",
            )
            ax.plot([], [], color=BASELINE, lw=1.2, label="Other learners")
            ax.set_title(f"{ARMS[arm]}: {outcome}")
            ax.xaxis.set_major_formatter(percent())
            ax.set_xlabel("Share of test customers targeted")
            ax.set_ylabel(f"Incremental {outcome}s per customer")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Qini curves on the held-out test split: above the line = better than random",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _learners(uplift: Any, training: Any, config: Config, path: Path) -> Path:
    outcomes = config.uplift.outcomes
    learners = config.uplift.learners
    fig, axes = new_figure(10.5, 2.9 * len(outcomes), len(outcomes), 2)
    cv = {(c["outcome"], c["arm"], c["learner"]): c for c in training["cross_validation"]}
    for row, outcome in enumerate(outcomes):
        for col, arm in enumerate(TREATMENT_ARMS):
            ax = axes[row * 2 + col]
            scores = {
                s["learner"]: s
                for s in uplift["scores"]
                if s["outcome"] == outcome and s["arm"] == arm
            }
            for i, learner in enumerate(learners):
                y = len(learners) - 1 - i
                q = scores[learner]["qini"]
                color = SERIES[0] if scores[learner]["selected"] else MUTED
                ax.hlines(y, q["low"], q["high"], color=color, lw=2)
                ax.scatter(
                    [q["estimate"]],
                    [y],
                    s=36,
                    color=color,
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=3,
                    label="Test (95% bootstrap CI)" if i == 0 else None,
                )
                ax.scatter(
                    [cv[(outcome, arm, learner)]["mean_qini"]],
                    [y],
                    s=36,
                    facecolor="none",
                    edgecolor=INK_SECONDARY,
                    linewidth=1.2,
                    zorder=4,
                    label="Training-split CV mean" if i == 0 else None,
                )
            ax.set_yticks(range(len(learners)), [LEARNER_LABELS[n] for n in reversed(learners)])
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{ARMS[arm]}: {outcome}")
            ax.set_xlabel("Qini coefficient (0 = random, 1 = oracle)")
            ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Learner comparison (blue = selected by training-split CV, before seeing the test split)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _calibration(uplift: Any, config: Config, path: Path) -> Path:
    outcome = config.policy.rank_outcome
    fig, axes = new_figure(10.5, 3.6, 1, 2)
    for ax, arm in zip(axes, TREATMENT_ARMS, strict=True):
        score = next(
            s
            for s in uplift["scores"]
            if s["outcome"] == outcome and s["arm"] == arm and s["selected"]
        )
        bins = score["calibration"]["bins"]
        predicted = np.array([b["predicted"] for b in bins])
        observed = np.array([b["observed"] for b in bins])
        se = np.array([b["se"] for b in bins])
        low = min(predicted.min(), (observed - 2 * se).min())
        high = max(predicted.max(), (observed + 2 * se).max())
        ax.plot([low, high], [low, high], color=MUTED, lw=1, label="Perfect calibration")
        ax.errorbar(
            predicted,
            observed,
            yerr=1.96 * se,
            fmt="o",
            color=ARM_COLORS[arm],
            ms=6,
            mec="white",
            mew=1.2,
            elinewidth=1.5,
            capsize=0,
            label="Decile of predicted uplift",
        )
        slope = score["calibration"]["slope"]
        ax.set_title(f"{ARMS[arm]} ({LEARNER_LABELS[score['learner']]}): slope {slope:.2f}")
        ax.set_xlabel(f"Predicted {outcome} uplift")
        ax.set_ylabel(f"Observed {outcome} uplift")
        ax.xaxis.set_major_formatter(percent())
        ax.yaxis.set_major_formatter(percent())
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Calibration by decile on the test split (whiskers: 95% CI of the observed uplift)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _shap(training: Any, shap: dict[str, np.ndarray], config: Config, path: Path) -> Path:
    outcome = config.policy.rank_outcome
    fig, axes = new_figure(11.5, 4.4, 1, 2)
    cmap_low, cmap_high = np.array([0xCD, 0xE2, 0xFB]) / 255, np.array([0x10, 0x42, 0x81]) / 255
    for ax, arm in zip(axes, TREATMENT_ARMS, strict=True):
        key = _panel_key(outcome, arm)
        if key not in training["shap"]:
            ax.set_visible(False)
            continue
        summary = training["shap"][key]
        values, features = shap[f"{key}/shap"], shap[f"{key}/features"]
        order = np.argsort(summary["mean_abs_shap"])[-8:]
        rng = np.random.default_rng(0)
        for row, j in enumerate(order):
            col = features[:, j]
            span = col.max() - col.min()
            t = (col - col.min()) / span if span > 0 else np.zeros_like(col)
            colors = cmap_low[None, :] * (1 - t[:, None]) + cmap_high[None, :] * t[:, None]
            jitter = rng.uniform(-0.28, 0.28, len(col))
            ax.scatter(values[:, j], row + jitter, s=5, c=colors, linewidths=0, rasterized=True)
        ax.set_yticks(range(len(order)), [FEATURE_LABELS[FEATURES[j]] for j in order])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:+.0f}")
        ax.set_xlabel(f"SHAP contribution to predicted {outcome} uplift (percentage points)")
        ax.set_title(f"{ARMS[arm]} ({LEARNER_LABELS[summary['learner']]})")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "What makes a customer persuadable (dot colour: feature value, light = low, dark = high)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _policy(policy: Any, config: Config, path: Path) -> Path:
    fig, axes = new_figure(12, 3.8, 1, 3)
    budget = config.policy.headline_budget
    labels = policy_labels(policy["rank_outcome"], ARMS[policy["best_single_arm"]])
    for ax, outcome in zip(axes, OUTCOMES, strict=True):
        curves = {c["policy"]: c for c in policy["curves"] if c["outcome"] == outcome}
        for name in ("random", "response", "uplift_single", "uplift_multi"):
            c = curves[name]
            x = np.array(c["budgets"])
            keep = x > 0
            y = np.array(c["incremental_per_1000"], dtype=float)[keep]
            if name == "uplift_multi":
                ax.fill_between(
                    x[keep],
                    np.array(c["incremental_per_1000_low"], dtype=float)[keep],
                    np.array(c["incremental_per_1000_high"], dtype=float)[keep],
                    color=POLICY_COLORS[name],
                    alpha=0.15,
                    lw=0,
                )
            ax.plot(
                x[keep],
                y,
                color=POLICY_COLORS[name],
                lw=2.2 if name == "uplift_multi" else 1.5,
                label=labels[name],
            )
        ax.axvline(budget, color=GRID, lw=1.5, zorder=0)
        ax.set_title(f"{PER_1000_LABELS[outcome]} per 1,000 emails")
        ax.set_xlabel("Budget: share of customers emailed")
        ax.xaxis.set_major_formatter(percent())
        ax.set_xlim(0.05, 1.0)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Off-policy value on the test split (SNIPW; band: 95% bootstrap CI of the uplift policy). "
        "Random targeting = emailing everyone, per email",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)
