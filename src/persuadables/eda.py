"""Exploratory figures of the raw experiment, written by ``persuadables data``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

from persuadables.config import OUTCOMES, Config
from persuadables.data.frame import ARMS, CONTROL_ARM, TREATMENT_ARMS, ExperimentData
from persuadables.report.style import (
    ARM_COLORS,
    BASELINE,
    FEATURE_LABELS,
    INK_SECONDARY,
    METRIC_LABELS,
    MUTED,
    new_figure,
    percent,
    save,
)
from persuadables.stats.design import balance_table


def write_eda_figures(data: ExperimentData, config: Config) -> list[Path]:
    out = config.output.figures_dir
    return [
        _outcomes_by_arm(data, out / "eda_outcomes.png"),
        _spend_distribution(data, out / "eda_spend.png"),
        _balance(data, config, out / "eda_balance.png"),
        _visit_by_recency(data, out / "eda_recency.png"),
    ]


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    half = (
        float(stats.t.isf(0.025, len(values) - 1))
        * float(values.std(ddof=1))
        / math.sqrt(len(values))
    )
    return mean, half


def _outcomes_by_arm(data: ExperimentData, path: Path) -> Path:
    fig, axes = new_figure(10.5, 3.2, 1, 3)
    for ax, metric in zip(axes, OUTCOMES, strict=True):
        y = data.outcome(metric)
        for position, arm in enumerate(ARMS):
            mean, half = _mean_ci(y[data.arm == arm])
            ax.bar(position, mean, width=0.55, color=ARM_COLORS[arm], zorder=2)
            ax.errorbar(position, mean, yerr=half, color=INK_SECONDARY, capsize=3, lw=1, zorder=3)
            label = f"${mean:.2f}" if metric == "spend" else f"{mean:.2%}"
            ax.annotate(
                label,
                (position, mean + half),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )
        ax.set_xticks(range(len(ARMS)), [ARMS[a] for a in ARMS])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
        if metric != "spend":
            ax.yaxis.set_major_formatter(percent())
        else:
            ax.yaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    fig.suptitle(
        "Outcomes by arm (bars: mean, whiskers: 95% CI; tests in the readout)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _spend_distribution(data: ExperimentData, path: Path) -> Path:
    spend = data.outcome("spend")
    zero_share = float((spend == 0).mean())
    paid = spend[spend > 0]
    fig, axes = new_figure(10.5, 3.4, 1, 2)
    bins = np.linspace(0, float(spend.max()), 41).tolist()
    for arm in ARMS:
        buyers = spend[(data.arm == arm) & (spend > 0)]
        axes[0].hist(
            buyers,
            bins=bins,
            histtype="step",
            lw=2,
            color=ARM_COLORS[arm],
            label=f"{ARMS[arm]} ({len(buyers)} buyers)",
        )
        ordered = np.sort(buyers)
        axes[1].plot(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            color=ARM_COLORS[arm],
            label=ARMS[arm],
        )
    axes[0].set_title("Spend among buyers")
    axes[0].set_xlabel("Spend ($)")
    axes[0].set_ylabel("Customers")
    axes[0].legend()
    axes[1].set_title("Cumulative share of buyers")
    axes[1].set_xlabel("Spend ($)")
    axes[1].yaxis.set_major_formatter(percent())
    fig.suptitle(
        f"Spend is zero for {zero_share:.1%} of customers (buyers: median ${np.median(paid):.0f}, "
        f"mean ${paid.mean():.0f}): hence a bootstrap CI next to Welch's t",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, path)


def _balance(data: ExperimentData, config: Config, path: Path) -> Path:
    names = config.experiment.balance_covariates
    rows = balance_table(
        data.arm, data.matrix(names), names, control=CONTROL_ARM, treatments=TREATMENT_ARMS
    )
    threshold = config.experiment.smd_threshold
    fig, axes = new_figure(7.5, 4.6)
    ax = axes[0]
    ax.axvspan(
        -threshold, threshold, color=BASELINE, alpha=0.25, lw=0, label=f"|SMD| < {threshold}"
    )
    ax.axvline(0, color=MUTED, lw=0.8)
    positions = {name: i for i, name in enumerate(reversed(names))}
    for offset, arm in zip((0.15, -0.15), TREATMENT_ARMS, strict=True):
        arm_rows = [r for r in rows if r.arm == arm]
        ax.scatter(
            [r.smd for r in arm_rows],
            [positions[r.covariate] + offset for r in arm_rows],
            s=36,
            color=ARM_COLORS[arm],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label=f"{ARMS[arm]} vs no email",
        )
    ax.set_yticks(range(len(names)), [FEATURE_LABELS.get(n, n) for n in reversed(names)])
    ax.set_xlim(-threshold * 1.5, threshold * 1.5)
    ax.set_xlabel("Standardized mean difference")
    worst = max(abs(r.smd) for r in rows)
    within = sum(abs(r.smd) < threshold for r in rows)
    ax.set_title(
        f"Covariate balance: {within} of {len(rows)} |SMD| below {threshold} (largest {worst:.3f})"
    )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def _visit_by_recency(data: ExperimentData, path: Path) -> Path:
    recency = data.numeric("recency")
    visit = data.outcome("visit")
    months = np.arange(1, 13)
    fig, axes = new_figure(7.5, 3.6)
    ax = axes[0]
    for arm in ARMS:
        in_arm = data.arm == arm
        rates = [float(visit[in_arm & (recency == m)].mean()) for m in months]
        ax.plot(
            months,
            rates,
            color=ARM_COLORS[arm],
            marker="o",
            ms=5,
            mec="white",
            mew=1.2,
            label=ARMS[arm],
        )
    ax.set_xticks(months)
    ax.set_xlabel("Months since last purchase")
    ax.yaxis.set_major_formatter(percent())
    ax.set_title("Visit rate by months since last purchase, per arm")
    ax.legend()
    fig.tight_layout()
    return save(fig, path)
