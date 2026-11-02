"""Apply the versioned SQL files in ``sql/`` to a fresh DuckDB database.

Each ``V###__name.sql`` file is one migration. They run in version order, each in its own
transaction, and their sha256 is recorded in ``schema_migrations`` so a database can always be
traced back to the exact SQL that built it. The database is rebuilt from scratch on every run:
the raw CSV is the source of truth and a rebuild takes about a second.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import duckdb

_MIGRATION_NAME = re.compile(r"^V(\d{3})__([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover_migrations() -> list[Migration]:
    """All packaged migrations in version order; versions must run 1, 2, 3, ... with no gaps."""
    migrations: list[Migration] = []
    for entry in resources.files("persuadables.data").joinpath("sql").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match is None:
            continue
        migrations.append(
            Migration(int(match.group(1)), match.group(2), entry.read_text(encoding="utf-8"))
        )
    migrations.sort(key=lambda m: m.version)
    expected = list(range(1, len(migrations) + 1))
    found = [m.version for m in migrations]
    if found != expected:
        msg = f"SQL migrations must be numbered 1..n without gaps, found versions {found}"
        raise ValueError(msg)
    return migrations


def build_database(
    raw_path: Path,
    database_path: Path,
    *,
    seed: int,
    test_fraction: float,
) -> list[Migration]:
    """Rebuild ``database_path`` from ``raw_path`` by applying every migration in order."""
    if not raw_path.is_file():
        msg = f"raw data not found at {raw_path}; run `persuadables data` to download it"
        raise FileNotFoundError(msg)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    database_path.with_name(database_path.name + ".wal").unlink(missing_ok=True)

    migrations = discover_migrations()
    with duckdb.connect(str(database_path)) as con:
        con.execute("SET VARIABLE raw_path = ?", [raw_path.as_posix()])
        con.execute("SET VARIABLE seed = ?", [seed])
        con.execute("SET VARIABLE test_tenths = ?", [round(test_fraction * 10)])
        con.execute(
            "CREATE TABLE schema_migrations ("
            "version INTEGER PRIMARY KEY, name VARCHAR NOT NULL, sha256 VARCHAR NOT NULL)"
        )
        for migration in migrations:
            con.begin()
            try:
                con.execute(migration.sql)
                con.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                    [migration.version, migration.name, migration.sha256],
                )
                con.commit()
            except duckdb.Error as error:
                con.rollback()
                msg = f"migration V{migration.version:03d}__{migration.name} failed: {error}"
                raise RuntimeError(msg) from error
    return migrations
