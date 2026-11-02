"""Download integrity, the SQL migrations, validation and the loader, on the fake dataset."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from persuadables.config import Config
from persuadables.data.download import DataIntegrityError, fetch_dataset, sha256_of
from persuadables.data.frame import FEATURES, load_experiment, segment_levels
from persuadables.data.pipeline import build_database, discover_migrations
from persuadables.data.validate import DataValidationError, raise_on_failure, validate_database
from tests.fakes import FAKE_ROWS, make_tiny_config, sha256_file, write_fake_hillstrom


def test_fetch_downloads_and_verifies(tmp_path: Path, fake_source: Path) -> None:
    destination = tmp_path / "raw" / "file.csv"
    fetched = fetch_dataset(fake_source.as_uri(), destination, sha256_file(fake_source))
    assert fetched == destination
    assert sha256_of(destination) == sha256_file(fake_source)
    # A verified copy is reused without touching the (now bogus) URL.
    assert (
        fetch_dataset("file:///does/not/exist.csv", destination, sha256_file(fake_source))
        == destination
    )


def test_fetch_discards_a_download_with_the_wrong_hash(tmp_path: Path, fake_source: Path) -> None:
    destination = tmp_path / "file.csv"
    with pytest.raises(DataIntegrityError, match="discarded"):
        fetch_dataset(fake_source.as_uri(), destination, "0" * 64)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_fetch_refuses_a_tampered_local_copy(tmp_path: Path, fake_source: Path) -> None:
    destination = tmp_path / "file.csv"
    destination.write_text("not the dataset")
    with pytest.raises(DataIntegrityError, match="Delete the file"):
        fetch_dataset(fake_source.as_uri(), destination, sha256_file(fake_source))


def test_migrations_are_numbered_without_gaps() -> None:
    migrations = discover_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert all(len(m.sha256) == 64 for m in migrations)


def test_pipeline_builds_a_traceable_database(built_workdir: tuple[Path, Config]) -> None:
    _, config = built_workdir
    with duckdb.connect(str(config.data.database_path), read_only=True) as con:
        recorded = con.execute(
            "SELECT version, sha256 FROM schema_migrations ORDER BY version"
        ).fetchall()
        levels = {r[0] for r in con.execute("SELECT DISTINCT zip_area FROM experiment").fetchall()}
    assert recorded == [(m.version, m.sha256) for m in discover_migrations()]
    assert levels == {"Urban", "Suburban", "Rural"}  # the source typo is fixed in V002


def test_split_is_stratified_and_deterministic(
    built_workdir: tuple[Path, Config], tmp_path: Path
) -> None:
    _, config = built_workdir
    data = load_experiment(config.data.database_path)
    for arm in (0, 1, 2):
        share = data.is_test[data.arm == arm].mean()
        assert share == pytest.approx(config.data.test_fraction, abs=0.01)
    rebuilt = tmp_path / "again.duckdb"
    build_database(
        config.data.raw_path, rebuilt, seed=config.seed, test_fraction=config.data.test_fraction
    )
    np.testing.assert_array_equal(load_experiment(rebuilt).is_test, data.is_test)


def test_loader_exposes_typed_columns(built_workdir: tuple[Path, Config]) -> None:
    _, config = built_workdir
    data = load_experiment(config.data.database_path)
    assert len(data) == FAKE_ROWS
    assert data.matrix(FEATURES).shape == (FAKE_ROWS, len(FEATURES))
    assert segment_levels(data.categorical("recency_band"), "recency_band")[0] == "1-3 months"
    with pytest.raises(TypeError, match="categorical"):
        data.numeric("channel")
    with pytest.raises(KeyError, match="unknown column"):
        data.numeric("nope")


def test_validation_passes_on_clean_data(built_workdir: tuple[Path, Config]) -> None:
    _, config = built_workdir
    report = validate_database(config.data.database_path, config)
    assert report.passed, report.failures


def _corrupt(source: Path, target: Path, line: int, column: int, value: str) -> None:
    lines = source.read_bytes().decode("ascii").split("\r\n")
    cells = lines[line].split(",")
    cells[column] = value
    lines[line] = ",".join(cells)
    target.write_bytes("\r\n".join(lines).encode("ascii"))


@pytest.mark.parametrize(
    ("column", "value", "failing_check"),
    [
        (0, "13", "recency is 1 to 12 months"),
        (8, "Kids E-Mail", "no nulls"),
        (10, "1", "a conversion implies a visit"),
    ],
)
def test_validation_names_the_broken_rule(
    tmp_path: Path, fake_source: Path, column: int, value: str, failing_check: str
) -> None:
    corrupted = tmp_path / "corrupted.csv"
    # A row without a quoted comma (so split(",") finds the columns) and with visit = 0 (so a
    # conversion there contradicts it).
    rows = fake_source.read_bytes().decode("ascii").split("\r\n")
    line = next(
        i for i, r in enumerate(rows[1:], start=1) if "$1,000" not in r and r.split(",")[9] == "0"
    )
    _corrupt(fake_source, corrupted, line, column, value)
    config = make_tiny_config(tmp_path, corrupted)
    build_database(corrupted, config.data.database_path, seed=1, test_fraction=0.3)
    report = validate_database(config.data.database_path, config)
    assert failing_check in {c.name for c in report.failures}
    with pytest.raises(DataValidationError, match=failing_check):
        raise_on_failure(report)


def test_validation_catches_a_wrong_row_count(tmp_path: Path) -> None:
    source = tmp_path / "small.csv"
    write_fake_hillstrom(source, 600, seed=3)
    config = make_tiny_config(tmp_path, source, rows=601)
    build_database(source, config.data.database_path, seed=1, test_fraction=0.3)
    failures = {c.name for c in validate_database(config.data.database_path, config).failures}
    assert "row count" in failures


@pytest.mark.parametrize(
    ("arm_shares", "break_balance", "failing_check"),
    [
        ((0.40, 0.30, 0.30), False, "no sample-ratio mismatch"),
        (None, True, "covariates balanced"),
    ],
)
def test_a_broken_randomization_stops_the_pipeline(
    tmp_path: Path,
    arm_shares: tuple[float, float, float] | None,
    break_balance: bool,
    failing_check: str,
) -> None:
    from persuadables.cli import run_data

    source = tmp_path / "remote" / "skewed.csv"
    write_fake_hillstrom(
        source, FAKE_ROWS, seed=11, arm_shares=arm_shares, break_balance=break_balance
    )
    config = make_tiny_config(tmp_path, source)
    with pytest.raises(DataValidationError, match=failing_check) as raised:
        run_data(config)
    # Only the randomization check fails: the rows themselves are valid.
    assert [c.name for c in raised.value.report.failures] == [failing_check]
    # The pipeline stopped before the EDA (and therefore before any analysis stage).
    assert not list(config.output.figures_dir.glob("eda_*.png"))


def test_build_fails_clearly_without_raw_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="persuadables data"):
        build_database(tmp_path / "missing.csv", tmp_path / "db.duckdb", seed=0, test_fraction=0.3)
