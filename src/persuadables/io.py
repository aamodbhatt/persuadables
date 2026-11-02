"""Deterministic seeds and JSON output that round-trips dataclasses, numpy values and NaN."""

from __future__ import annotations

import dataclasses
import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np


def derive_seed(base: int, *keys: int | str) -> int:
    """A stable 32-bit seed for one named stochastic step, derived from the run seed.

    Separate streams per step mean adding a bootstrap somewhere does not shift every other
    random number in the run.
    """
    entropy = [base] + [k if isinstance(k, int) else zlib.crc32(k.encode("utf-8")) for k in keys]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, tuples, numpy scalars/arrays and paths to JSON-ready values.

    Non-finite floats become ``null`` so the files stay valid JSON.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(value), indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    if not path.is_file():
        msg = f"{path} not found; run the command that produces it first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))
