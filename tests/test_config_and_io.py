"""Typed configuration and the JSON/seed helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from persuadables.config import load_config
from persuadables.io import derive_seed, read_json, to_jsonable, write_json
from tests.fakes import ROOT


def test_default_config_loads_and_anchors_paths(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "default.toml", root=tmp_path)
    assert config.data.raw_path == tmp_path / "data" / "raw" / "hillstrom.csv"
    assert config.policy.headline_budget in config.policy.budgets


def test_config_rejects_bad_values(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("test_fraction = 0.3", "test_fraction = 0.25"), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"multiple of 0\.1"):
        load_config(bad)
    bad.write_text(text.replace("headline_budget = 0.3", "headline_budget = 0.2"), encoding="utf-8")
    with pytest.raises(ValidationError, match="headline_budget"):
        load_config(bad)
    bad.write_text(text + "\nunknown_key = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_missing_config_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "nope.toml")


def test_derive_seed_is_stable_and_separates_streams() -> None:
    assert derive_seed(1, "cv", 2) == derive_seed(1, "cv", 2)
    assert len({derive_seed(1, "cv", k) for k in range(100)}) == 100
    assert derive_seed(1, "cv") != derive_seed(2, "cv")


def test_json_round_trip_handles_numpy_and_nan(tmp_path: Path) -> None:
    value = {"a": np.float64(1.5), "b": (np.int64(2), math.nan), "c": np.array([1.0, math.inf])}
    path = tmp_path / "out.json"
    write_json(path, value)
    assert read_json(path) == {"a": 1.5, "b": [2, None], "c": [1.0, None]}
    assert to_jsonable(Path("a/b")) == "a/b"
    json.loads(path.read_text(encoding="utf-8"))
