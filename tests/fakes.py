"""Test helpers: a small fake MineThatData file with the real schema, and a tiny config.

The default suite never touches the network or the real dataset: the fake CSV has the same
columns, levels and quirks (CRLF line endings, the "Surburban" typo) and a planted treatment
effect, so every stage runs end to end in seconds.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from persuadables.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
FAKE_ROWS = 6000

_HISTORY_SEGMENTS = (
    (100, "1) $0 - $100"),
    (200, "2) $100 - $200"),
    (350, "3) $200 - $350"),
    (500, "4) $350 - $500"),
    (750, "5) $500 - $750"),
    (1000, "6) $750 - $1,000"),
)


def _history_segment(value: float) -> str:
    for upper, label in _HISTORY_SEGMENTS:
        if value < upper:
            return label
    return "7) $1,000 +"


def write_fake_hillstrom(
    path: Path,
    n: int,
    seed: int,
    *,
    arm_shares: tuple[float, float, float] | None = None,
    break_balance: bool = False,
) -> None:
    """Write a fake source file. ``arm_shares`` (no email, Mens, Womens) skews assignment away
    from the 1/3 design (a sample-ratio mismatch); ``break_balance`` makes every Mens-email
    customer a newbie, a covariate imbalance no randomization produces."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 13, n)
    history = np.round(29.99 + rng.lognormal(4.5, 1.0, n), 2)
    kind = rng.integers(0, 3, n)  # mens only, womens only, both
    mens = (kind != 1).astype(int)
    womens = (kind != 0).astype(int)
    zip_code = rng.choice(["Urban", "Surburban", "Rural"], n)
    newbie = rng.integers(0, 2, n)
    channel = rng.choice(["Phone", "Web", "Multichannel"], n)
    arms = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
    # The default path keeps its own draw so the shared fake file never changes.
    segment = rng.choice(arms, n) if arm_shares is None else rng.choice(arms, n, p=arm_shares)
    if break_balance:
        newbie = np.where(segment == "Mens E-Mail", 1, newbie)
    base = 0.08 + 0.10 * (recency <= 3)
    lift = np.where(segment == "Mens E-Mail", 0.08 * mens, 0.0) + np.where(
        segment == "Womens E-Mail", 0.06 * womens, 0.0
    )
    visit = (rng.random(n) < base + lift).astype(int)
    conversion = visit * (rng.random(n) < 0.12).astype(int)
    spend = np.where(conversion == 1, np.round(rng.uniform(29.99, 499.0, n), 2), 0.0)
    header = (
        "recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,"
        "visit,conversion,spend"
    )
    lines = [header] + [
        f'{recency[i]},"{_history_segment(history[i])}",{history[i]},{mens[i]},{womens[i]},'
        f"{zip_code[i]},{newbie[i]},{channel[i]},{segment[i]},{visit[i]},{conversion[i]},"
        f"{spend[i]:g}"
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tiny_config(workdir: Path, source: Path, rows: int = FAKE_ROWS) -> Config:
    """The default config, shrunk for speed and pointed at ``source`` and ``workdir``."""
    base = load_config(ROOT / "configs" / "default.toml", root=workdir)
    tiny_xgb = base.uplift.xgboost.model_copy(update={"n_estimators": 15, "n_jobs": 2})
    return base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "url": source.resolve().as_uri(),
                    "sha256": sha256_file(source),
                    "expected_rows": rows,
                }
            ),
            "experiment": base.experiment.model_copy(update={"bootstrap_resamples": 200}),
            "uplift": base.uplift.model_copy(
                update={"cv_folds": 2, "bootstrap_resamples": 20, "xgboost": tiny_xgb}
            ),
            "policy": base.policy.model_copy(update={"grid_step": 0.1, "bootstrap_resamples": 60}),
            "output": base.output.model_copy(update={"plotly_js": "cdn"}),
        }
    )


def write_tiny_toml(path: Path, source: Path, rows: int = FAKE_ROWS) -> Path:
    """The default TOML with the same shrinking as ``make_tiny_config``, for CLI tests."""
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    text = re.sub(r'(?m)^url = ".*"$', f'url = "{source.resolve().as_uri()}"', text)
    # Order matters: the uplift "= 200" line shrinks before the experiment's "= 2000" becomes 200.
    replacements = {
        "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece": sha256_file(source),
        "expected_rows = 64000": f"expected_rows = {rows}",
        "bootstrap_resamples = 200\n": "bootstrap_resamples = 20\n",
        "bootstrap_resamples = 2000": "bootstrap_resamples = 200",
        "cv_folds = 5": "cv_folds = 2",
        "n_estimators = 200": "n_estimators = 15",
        "n_jobs = 4": "n_jobs = 2",
        "grid_step = 0.01": "grid_step = 0.1",
        "bootstrap_resamples = 1000": "bootstrap_resamples = 60",
        'plotly_js = "inline"': 'plotly_js = "cdn"',
    }
    for old, new in replacements.items():
        if old not in text:
            msg = f"default.toml no longer contains {old!r}; update write_tiny_toml"
            raise AssertionError(msg)
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return path
