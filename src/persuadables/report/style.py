"""One visual system for the static figures and the dashboard.

Colours are assigned by job: a fixed categorical order for identities (arms, policies), a neutral
grey for references (control, random targeting), and ink tokens for text. The categorical steps
were checked for colour-vision-deficiency separation as an ordered set, so slots are used in order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

ARM_COLORS = {0: MUTED, 1: SERIES[0], 2: SERIES[1]}
POLICY_COLORS = {
    "uplift_multi": SERIES[0],
    "response": SERIES[1],
    "uplift_single": SERIES[2],
    "random": MUTED,
}
POLICY_LABELS = {
    "uplift_multi": "Uplift model, best email per customer",
    "uplift_single": "Uplift model, one email for all",
    "response": "Response model",
    "random": "Random targeting",
    "no_one": "Email no one",
    "everyone_arm1": "Email everyone (Mens)",
    "everyone_arm2": "Email everyone (Womens)",
}


def policy_labels(rank_outcome: str, best_arm_name: str) -> dict[str, str]:
    """Policy display names. The response baseline is named by what it actually scores, the
    probability of the ranking outcome under the single best email, so no reader mistakes a
    visit model for a purchase model."""
    return {**POLICY_LABELS, "response": f"Response model: P({rank_outcome} | {best_arm_name})"}


METRIC_LABELS = {
    "visit": "Visit rate",
    "conversion": "Conversion rate",
    "spend": "Spend per customer",
}
FEATURE_LABELS = {
    "recency": "Months since last purchase",
    "history": "Spend in past year ($)",
    "mens": "Bought mens merchandise",
    "womens": "Bought womens merchandise",
    "newbie": "New customer",
    "zip_urban": "Urban",
    "zip_suburban": "Suburban",
    "zip_rural": "Rural",
    "channel_phone": "Bought by phone",
    "channel_web": "Bought on the web",
    "channel_multichannel": "Bought multichannel",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            # Dollar signs in labels ("$200-$500") are text, not math.
            "text.parse_math": False,
        }
    )


def percent() -> PercentFormatter:
    """Percent ticks whose decimals follow the axis range (2.5% never shows as "2%")."""
    return PercentFormatter(1.0)


def new_figure(
    width: float, height: float, nrows: int = 1, ncols: int = 1
) -> tuple[Figure, list[Axes]]:
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    return fig, [ax for row in axes for ax in row]


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed timestamp-free PNG keeps reruns byte-stable for git.
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path
