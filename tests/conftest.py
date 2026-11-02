"""Shared fixtures. The default suite never touches the network or the real dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from persuadables.config import Config
from tests.fakes import FAKE_ROWS, make_tiny_config, write_fake_hillstrom

os.environ.setdefault("OMP_NUM_THREADS", "2")


@pytest.fixture(scope="session")
def fake_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("remote") / "hillstrom.csv"
    write_fake_hillstrom(path, FAKE_ROWS, seed=7)
    return path


@pytest.fixture(scope="session")
def built_workdir(
    tmp_path_factory: pytest.TempPathFactory, fake_source: Path
) -> tuple[Path, Config]:
    """A workdir with the fake data downloaded, built and validated (shared, read-only)."""
    from persuadables.cli import run_data

    workdir = tmp_path_factory.mktemp("work")
    config = make_tiny_config(workdir, fake_source)
    run_data(config)
    return workdir, config
