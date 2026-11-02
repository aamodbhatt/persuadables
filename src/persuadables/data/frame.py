"""The in-memory view of the experiment: plain numpy arrays loaded from the DuckDB ``experiment``
table. Every analysis module takes an ``ExperimentData`` and nothing else."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from persuadables._types import AnyArray, BoolArray, FloatArray, IntArray, StrArray
from persuadables.config import Outcome

#: Treatment arms as coded by migration V002. Arm 0 is the holdout (control).
ARMS: Mapping[int, str] = {0: "No email", 1: "Mens email", 2: "Womens email"}
CONTROL_ARM = 0
TREATMENT_ARMS: tuple[int, ...] = (1, 2)

#: Pre-period model inputs (all measured before randomization).
FEATURES: tuple[str, ...] = (
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_urban",
    "zip_suburban",
    "zip_rural",
    "channel_phone",
    "channel_web",
    "channel_multichannel",
)

#: Display order of each pre-registered segment's levels (defined in migration V003).
SEGMENT_LEVELS: Mapping[str, tuple[str, ...]] = {
    "purchase_category": ("mens only", "womens only", "both", "neither"),
    "recency_band": ("1-3 months", "4-6 months", "7-9 months", "10-12 months"),
    "history_tier": ("under $200", "$200-$500", "$500 and over"),
    "channel": ("Phone", "Web", "Multichannel"),
    "zip_area": ("Urban", "Suburban", "Rural"),
    "tenure": ("new", "returning"),
}

_STRING_COLUMNS = frozenset(SEGMENT_LEVELS)


@dataclass(frozen=True)
class ExperimentData:
    """Columns of the ``experiment`` table, one array per column, all the same length."""

    columns: Mapping[str, AnyArray]

    def __post_init__(self) -> None:
        lengths = {name: len(values) for name, values in self.columns.items()}
        if len(set(lengths.values())) > 1:
            msg = f"columns have different lengths: {lengths}"
            raise ValueError(msg)
        missing = {"customer_id", "arm", "visit", "conversion", "spend", "is_test"} - set(lengths)
        if missing:
            msg = f"experiment data is missing required columns: {sorted(missing)}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.columns["arm"])

    @property
    def arm(self) -> IntArray:
        return self.columns["arm"].astype(np.int64)

    @property
    def is_test(self) -> BoolArray:
        return self.columns["is_test"].astype(np.bool_)

    def outcome(self, name: Outcome) -> FloatArray:
        return self.numeric(name)

    def numeric(self, name: str) -> FloatArray:
        if name not in self.columns:
            msg = f"unknown column {name!r}; available: {sorted(self.columns)}"
            raise KeyError(msg)
        if name in _STRING_COLUMNS:
            msg = f"column {name!r} is categorical, not numeric"
            raise TypeError(msg)
        return self.columns[name].astype(np.float64)

    def matrix(self, names: tuple[str, ...]) -> FloatArray:
        """Stack numeric columns into an (n, len(names)) float matrix."""
        return np.column_stack([self.numeric(name) for name in names])

    def categorical(self, name: str) -> StrArray:
        if name not in _STRING_COLUMNS:
            choices = sorted(_STRING_COLUMNS)
            msg = f"column {name!r} is not a categorical segment; choose from {choices}"
            raise KeyError(msg)
        return self.columns[name].astype(np.str_)

    def subset(self, mask: BoolArray) -> ExperimentData:
        return ExperimentData({name: values[mask] for name, values in self.columns.items()})


def load_experiment(database_path: Path) -> ExperimentData:
    """Read the ``experiment`` table built by the SQL pipeline, ordered by customer_id."""
    if not database_path.is_file():
        msg = f"database not found at {database_path}; run `persuadables data` first"
        raise FileNotFoundError(msg)
    with duckdb.connect(str(database_path), read_only=True) as con:
        raw = con.execute("SELECT * FROM experiment ORDER BY customer_id").fetchnumpy()
    return ExperimentData({name: np.asarray(values) for name, values in raw.items()})


def segment_levels(values: StrArray, segment: str) -> tuple[str, ...]:
    """Levels present in ``values``, in the declared display order (unknown levels last, sorted)."""
    present = set(np.unique(values).tolist())
    declared = SEGMENT_LEVELS.get(segment, ())
    return tuple(level for level in declared if level in present) + tuple(
        sorted(present - set(declared))
    )
