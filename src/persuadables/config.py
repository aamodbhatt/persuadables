"""Typed run configuration, loaded from a TOML file and validated at the boundary.

Every stochastic step (data split, cross-validation folds, model fitting, bootstraps) derives
its seed from ``Config.seed``, so one config file pins every number in ``results/``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Outcome = Literal["visit", "conversion", "spend"]
BinaryOutcome = Literal["visit", "conversion"]
LearnerName = Literal["s", "t", "x", "dr", "ct"]

#: Outcomes in the order every report lists them.
OUTCOMES: tuple[Outcome, ...] = ("visit", "conversion", "spend")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Where the raw data comes from and where the pipeline writes it."""

    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_path: Path
    database_path: Path
    expected_rows: int = Field(gt=0)
    #: Share of customers held out for the uplift and policy evaluation. A multiple of 0.1
    #: because the SQL split assigns rows to folds of ten within each stratum.
    test_fraction: float = Field(gt=0.0, lt=1.0)

    @field_validator("test_fraction")
    @classmethod
    def _tenths(cls, value: float) -> float:
        if abs(value * 10 - round(value * 10)) > 1e-9:
            msg = f"test_fraction must be a multiple of 0.1, got {value}"
            raise ValueError(msg)
        return value


class ExperimentConfig(_Strict):
    """Settings of the A/B readout. Segments and covariates are fixed before any effect is
    estimated."""

    alpha: float = Field(gt=0.0, lt=0.5)
    power: float = Field(gt=0.5, lt=1.0)
    bootstrap_resamples: int = Field(ge=100)
    #: Pre-period columns of the ``features`` table used for CUPED.
    cuped_covariates: tuple[str, ...] = Field(min_length=1)
    #: Pre-registered segment columns of the ``features`` table.
    segments: tuple[str, ...] = Field(min_length=1)
    #: Covariates checked for balance (standardized mean differences).
    balance_covariates: tuple[str, ...] = Field(min_length=1)
    #: |SMD| above this flags imbalance; 0.1 is the conventional threshold.
    smd_threshold: float = Field(gt=0.0)
    #: SRM p-values below this fail the data validation (a broken randomizer, not bad luck).
    srm_alpha: float = Field(gt=0.0, lt=0.5)
    #: A planning-sized relative lift (0.1 = +10% over the control mean). The readout reports
    #: the power the realised arms had for it and the arm size it would need; it changes no
    #: estimate.
    reference_relative_lift: float = Field(gt=0.0)


class XGBoostConfig(_Strict):
    """Hyperparameters shared by every base model. Shallow, regularized trees on purpose:
    uplift is a small signal on top of a large baseline, and deep trees fit the baseline noise."""

    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1, le=12)
    learning_rate: float = Field(gt=0.0, le=1.0)
    subsample: float = Field(gt=0.0, le=1.0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    min_child_weight: float = Field(ge=0.0)
    reg_lambda: float = Field(ge=0.0)
    n_jobs: int = Field(ge=1)


class UpliftConfig(_Strict):
    outcomes: tuple[BinaryOutcome, ...] = Field(min_length=1)
    learners: tuple[LearnerName, ...] = Field(min_length=1)
    cv_folds: int = Field(ge=2)
    #: Internal cross-fitting folds of the DR-learner's nuisance models.
    dr_crossfit_folds: int = Field(ge=2)
    #: Fractions at which uplift@k is reported.
    uplift_at: tuple[float, ...] = Field(min_length=1)
    calibration_bins: int = Field(ge=2)
    #: Resamples for the bootstrap interval of each test-split Qini coefficient.
    bootstrap_resamples: int = Field(ge=20)
    xgboost: XGBoostConfig


class PolicyConfig(_Strict):
    #: Outcome whose uplift model ranks customers.
    rank_outcome: BinaryOutcome
    #: Budget (share of customers emailed) quoted in the headline; fixed before evaluation.
    headline_budget: float = Field(gt=0.0, le=1.0)
    budgets: tuple[float, ...] = Field(min_length=1)
    #: Step of the budget grid used for the policy curves and the simulator.
    grid_step: float = Field(gt=0.0, le=0.5)
    bootstrap_resamples: int = Field(ge=50)


class OutputConfig(_Strict):
    artifacts_dir: Path
    results_dir: Path
    figures_dir: Path
    report_path: Path
    #: "inline" embeds plotly.js so the dashboard works offline; "cdn" keeps the file small.
    plotly_js: Literal["inline", "cdn"]


class Config(_Strict):
    seed: int = Field(ge=0)
    data: DataConfig
    experiment: ExperimentConfig
    uplift: UpliftConfig
    policy: PolicyConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _headline_in_budgets(self) -> Config:
        if self.policy.headline_budget not in self.policy.budgets:
            msg = "policy.headline_budget must be one of policy.budgets"
            raise ValueError(msg)
        return self

    def resolve(self, root: Path) -> Config:
        """Return a copy with every relative path anchored at ``root``."""

        def anchor(path: Path) -> Path:
            return path if path.is_absolute() else root / path

        data = self.data.model_copy(
            update={
                "raw_path": anchor(self.data.raw_path),
                "database_path": anchor(self.data.database_path),
            }
        )
        output = self.output.model_copy(
            update={
                "artifacts_dir": anchor(self.output.artifacts_dir),
                "results_dir": anchor(self.output.results_dir),
                "figures_dir": anchor(self.output.figures_dir),
                "report_path": anchor(self.output.report_path),
            }
        )
        return self.model_copy(update={"data": data, "output": output})


def load_config(path: Path, root: Path | None = None) -> Config:
    """Load and validate a TOML config; relative paths resolve against ``root`` (default: cwd)."""
    if not path.is_file():
        msg = f"config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return Config.model_validate(raw).resolve(root or Path.cwd())
