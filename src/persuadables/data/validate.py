"""Data validation: schema, nulls, value ranges, logical consistency, the split, and whether the
randomization looks like a randomization.

Row-level rules are SQL predicates evaluated in DuckDB, so a failure reports how many rows break
the rule and shows a few of them. The pipeline refuses to continue on any failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from persuadables.config import Config
from persuadables.data.frame import ARMS, CONTROL_ARM, TREATMENT_ARMS, load_experiment
from persuadables.stats.design import balance_table, sample_ratio_mismatch

EXPECTED_SCHEMA: dict[str, str] = {
    "customer_id": "INTEGER",
    "arm": "INTEGER",
    "visit": "INTEGER",
    "conversion": "INTEGER",
    "spend": "DOUBLE",
    "recency": "INTEGER",
    "history": "DOUBLE",
    "log_history": "DOUBLE",
    "mens": "INTEGER",
    "womens": "INTEGER",
    "newbie": "INTEGER",
    "zip_urban": "INTEGER",
    "zip_suburban": "INTEGER",
    "zip_rural": "INTEGER",
    "channel_phone": "INTEGER",
    "channel_web": "INTEGER",
    "channel_multichannel": "INTEGER",
    "purchase_category": "VARCHAR",
    "recency_band": "VARCHAR",
    "history_tier": "VARCHAR",
    "channel": "VARCHAR",
    "zip_area": "VARCHAR",
    "tenure": "VARCHAR",
    "is_test": "BOOLEAN",
}

#: (rule, SQL predicate every row must satisfy). NULL never satisfies a predicate.
ROW_RULES: tuple[tuple[str, str], ...] = (
    ("arm is 0, 1 or 2", "arm IN (0, 1, 2)"),
    ("recency is 1 to 12 months", "recency BETWEEN 1 AND 12"),
    ("history is positive", "history > 0"),
    ("spend is non-negative", "spend >= 0"),
    (
        "flags are 0 or 1",
        "visit IN (0, 1) AND conversion IN (0, 1) AND mens IN (0, 1) AND womens IN (0, 1) "
        "AND newbie IN (0, 1)",
    ),
    ("zip_area is a known level", "zip_area IN ('Urban', 'Suburban', 'Rural')"),
    ("channel is a known level", "channel IN ('Phone', 'Web', 'Multichannel')"),
    ("a conversion implies a visit", "conversion <= visit"),
    ("spend is positive exactly when the customer converted", "(spend > 0) = (conversion = 1)"),
    (
        "exactly one zip and one channel indicator",
        "zip_urban + zip_suburban + zip_rural = 1 "
        "AND channel_phone + channel_web + channel_multichannel = 1",
    ),
)

#: Tolerance on each arm's realised test share around the configured test fraction.
_SPLIT_TOLERANCE = 0.01


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        lines = [f"- {c.name}: {c.detail}" for c in report.failures]
        super().__init__("data validation failed:\n" + "\n".join(lines))
        self.report = report


def validate_database(database_path: Path, config: Config) -> ValidationReport:
    """Run every check against the ``experiment`` table and return the report (never raises on
    a failed check; call ``raise_on_failure`` for that)."""
    checks: list[CheckResult] = []
    with duckdb.connect(str(database_path), read_only=True) as con:
        checks.append(_check_schema(con))
        if not checks[-1].passed:
            # Later checks would fail with confusing SQL errors on a wrong schema.
            return ValidationReport(tuple(checks))
        checks.append(_check_row_count(con, config.data.expected_rows))
        checks.append(_check_nulls(con))
        checks.append(_check_unique_ids(con))
        checks.extend(_check_row_rule(con, name, predicate) for name, predicate in ROW_RULES)
        checks.append(_check_split(con, config.data.test_fraction))
    if all(check.passed for check in checks):
        checks.extend(_check_randomization(database_path, config))
    else:
        # SRM and balance statistics on rows that break the rules above would be meaningless.
        checks.append(
            CheckResult("randomization", False, "not checked: fix the failures above first")
        )
    return ValidationReport(tuple(checks))


def raise_on_failure(report: ValidationReport) -> None:
    if not report.passed:
        raise DataValidationError(report)


def _check_schema(con: duckdb.DuckDBPyConnection) -> CheckResult:
    actual = {str(row[0]): str(row[1]) for row in con.execute("DESCRIBE experiment").fetchall()}
    problems = [
        f"{column} is {actual.get(column, 'missing')}, expected {expected}"
        for column, expected in EXPECTED_SCHEMA.items()
        if actual.get(column) != expected
    ]
    extra = sorted(set(actual) - set(EXPECTED_SCHEMA))
    if extra:
        problems.append(f"unexpected columns {extra}")
    return CheckResult(
        "schema", not problems, "; ".join(problems) or f"{len(actual)} typed columns"
    )


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def _check_row_count(con: duckdb.DuckDBPyConnection, expected: int) -> CheckResult:
    rows = _scalar(con, "SELECT count(*) FROM experiment")
    return CheckResult("row count", rows == expected, f"{rows:,} rows (expected {expected:,})")


def _check_nulls(con: duckdb.DuckDBPyConnection) -> CheckResult:
    columns = list(EXPECTED_SCHEMA)
    counts = con.execute(
        "SELECT " + ", ".join(f'count_if("{c}" IS NULL)' for c in columns) + " FROM experiment"
    ).fetchone()
    assert counts is not None
    nulls = {c: int(n) for c, n in zip(columns, counts, strict=True) if n}
    detail = f"NULLs in {nulls}" if nulls else "no NULLs in any column"
    return CheckResult("no nulls", not nulls, detail)


def _check_unique_ids(con: duckdb.DuckDBPyConnection) -> CheckResult:
    duplicates = _scalar(con, "SELECT count(*) - count(DISTINCT customer_id) FROM experiment")
    return CheckResult("unique customer_id", duplicates == 0, f"{duplicates} duplicate ids")


def _check_row_rule(con: duckdb.DuckDBPyConnection, name: str, predicate: str) -> CheckResult:
    violating = f"FROM experiment WHERE NOT coalesce(({predicate}), false)"
    count = _scalar(con, f"SELECT count(*) {violating}")
    if count == 0:
        return CheckResult(name, True, "all rows pass")
    examples = [
        int(r[0]) for r in con.execute(f"SELECT customer_id {violating} LIMIT 5").fetchall()
    ]
    return CheckResult(
        name, False, f"{count} rows violate `{predicate}`, e.g. customer_id {examples}"
    )


def _check_split(con: duckdb.DuckDBPyConnection, test_fraction: float) -> CheckResult:
    shares = con.execute(
        "SELECT arm, avg(CAST(is_test AS DOUBLE)) FROM experiment "
        "WHERE arm IS NOT NULL GROUP BY arm ORDER BY arm"
    ).fetchall()
    off = {
        int(arm): round(float(s), 4)
        for arm, s in shares
        if abs(s - test_fraction) > _SPLIT_TOLERANCE
    }
    detail = (
        f"test share per arm {off} is off target {test_fraction}"
        if off
        else f"test share within {_SPLIT_TOLERANCE} of {test_fraction} in every arm"
    )
    return CheckResult("stratified split", not off, detail)


def _check_randomization(database_path: Path, config: Config) -> list[CheckResult]:
    data = load_experiment(database_path)
    arm = data.arm
    counts = tuple(int((arm == a).sum()) for a in ARMS)
    srm = sample_ratio_mismatch(counts, tuple(1 / len(ARMS) for _ in ARMS))
    srm_ok = srm.p_value >= config.experiment.srm_alpha
    rows = balance_table(
        arm,
        data.matrix(config.experiment.balance_covariates),
        config.experiment.balance_covariates,
        control=CONTROL_ARM,
        treatments=TREATMENT_ARMS,
    )
    worst = max(rows, key=lambda r: abs(r.smd))
    balance_ok = abs(worst.smd) < config.experiment.smd_threshold
    return [
        CheckResult(
            "no sample-ratio mismatch",
            srm_ok,
            f"arm sizes {counts}, chi2={srm.chi2:.3f}, p={srm.p_value:.4f} "
            f"(fails below {config.experiment.srm_alpha})",
        ),
        CheckResult(
            "covariates balanced",
            balance_ok,
            f"largest |SMD| is {abs(worst.smd):.4f} ({worst.covariate}, {ARMS[worst.arm]}); "
            f"threshold {config.experiment.smd_threshold}",
        ),
    ]
