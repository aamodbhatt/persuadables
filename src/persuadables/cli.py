"""Command-line interface: ``persuadables data | analyze | train | evaluate | report | all``.

Each stage reads the previous stage's outputs from disk, so any stage can be rerun alone.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from persuadables.config import Config, load_config

app = typer.Typer(
    help="Who does a marketing campaign actually persuade? Experiment readout, uplift models and "
    "a budgeted targeting policy on the MineThatData e-mail experiment.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="TOML run configuration.", exists=True, dir_okay=False),
]
DEFAULT_CONFIG = Path("configs/default.toml")


def _load(config_path: Path) -> Config:
    # Cap BLAS/OpenMP pools before numpy or XGBoost start them: the run stays laptop-friendly.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "4")
    return load_config(config_path)


def _say(stage: str, message: str) -> None:
    typer.echo(f"[{stage}] {message}")


def run_data(config: Config) -> None:
    from persuadables.data.download import fetch_dataset
    from persuadables.data.frame import load_experiment
    from persuadables.data.pipeline import build_database
    from persuadables.data.validate import raise_on_failure, validate_database
    from persuadables.eda import write_eda_figures
    from persuadables.io import write_json

    cfg = config.data
    raw = fetch_dataset(cfg.url, cfg.raw_path, cfg.sha256)
    _say("data", f"raw file verified: {raw} (sha256 {cfg.sha256[:12]}...)")
    migrations = build_database(
        raw, cfg.database_path, seed=config.seed, test_fraction=cfg.test_fraction
    )
    _say("data", f"applied {len(migrations)} SQL migrations -> {cfg.database_path}")
    report = validate_database(cfg.database_path, config)
    write_json(
        config.output.results_dir / "data_validation.json",
        {
            "passed": report.passed,
            "checks": report.checks,
            "migrations": [
                {"version": m.version, "name": m.name, "sha256": m.sha256} for m in migrations
            ],
        },
    )
    for check in report.checks:
        _say("data", f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    raise_on_failure(report)
    figures = write_eda_figures(load_experiment(cfg.database_path), config)
    _say("data", f"wrote {len(figures)} EDA figures to {config.output.figures_dir}")


def run_analyze(config: Config) -> None:
    from persuadables.data.frame import load_experiment
    from persuadables.experiment import run_readout
    from persuadables.io import write_json

    started = time.perf_counter()
    readout = run_readout(load_experiment(config.data.database_path), config)
    write_json(config.output.results_dir / "readout.json", readout)
    for effect in readout.effects:
        e = effect.estimate
        _say(
            "analyze",
            f"{effect.arm_name:<13} {effect.metric:<10} diff={e.diff:+.5f} "
            f"[{e.ci_low:+.5f}, {e.ci_high:+.5f}] p_holm={effect.p_holm:.2e} "
            f"CUPED variance reduction={effect.cuped.variance_reduction:.2%}",
        )
    _say("analyze", f"done in {time.perf_counter() - started:.1f}s")


def run_train(config: Config) -> None:
    from persuadables.data.frame import load_experiment
    from persuadables.io import write_json
    from persuadables.training import (
        save_predictions,
        train,
        training_summary,
        training_timings,
    )

    result = train(load_experiment(config.data.database_path), config)
    save_predictions(config.output.artifacts_dir / "predictions.npz", result.test_arrays)
    save_predictions(config.output.artifacts_dir / "shap_sample.npz", result.shap_arrays())
    write_json(config.output.results_dir / "training.json", training_summary(result))
    write_json(config.output.artifacts_dir / "timings_train.json", training_timings(result))
    for cv in result.cv:
        _say(
            "train",
            f"{cv.outcome:<10} arm {cv.arm} {cv.learner:<3} CV Qini "
            f"{cv.mean_qini:+.4f} +/- {cv.sd_qini:.4f} ({cv.seconds:.1f}s)",
        )
    _say("train", f"selected learners: {result.selected}; done in {result.seconds:.1f}s")


def run_evaluate(config: Config) -> None:
    from persuadables.evaluation import evaluate
    from persuadables.io import read_json, write_json
    from persuadables.training import load_predictions

    started = time.perf_counter()
    test = load_predictions(config.output.artifacts_dir / "predictions.npz")
    summary: dict[str, Any] = read_json(config.output.results_dir / "training.json")
    best_arm = {k: int(v) for k, v in summary["best_single_arm"].items()}
    result = evaluate(test, summary["selected"], best_arm, config)
    write_json(config.output.results_dir / "uplift_test.json", result.uplift)
    write_json(config.output.results_dir / "policy.json", result.policy)
    for score in result.uplift.scores:
        mark = "*" if score.selected else " "
        _say(
            "evaluate",
            f"{mark} {score.outcome:<10} arm {score.arm} {score.learner:<3} test Qini "
            f"{score.qini.estimate:+.4f} [{score.qini.low:+.4f}, {score.qini.high:+.4f}]",
        )
    _say("evaluate", f"done in {time.perf_counter() - started:.1f}s")


def run_report(config: Config) -> None:
    from persuadables.report.dashboard import write_dashboard
    from persuadables.report.figures import write_result_figures

    figures = write_result_figures(config)
    _say("report", f"wrote {len(figures)} result figures to {config.output.figures_dir}")
    path = write_dashboard(config)
    _say("report", f"dashboard: {path}")


@app.command()
def data(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Download (sha256-verified), build the DuckDB feature tables, validate, and plot the EDA."""
    run_data(_load(config))


@app.command()
def analyze(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Experiment readout: SRM, balance, effects, CUPED, Holm, power, segment heterogeneity."""
    run_analyze(_load(config))


@app.command(name="train")
def train_command(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Cross-validate the uplift learners, select, refit and score the test split."""
    run_train(_load(config))


@app.command(name="evaluate")
def evaluate_command(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Test-split Qini/AUUC/uplift@k/calibration and off-policy values of targeting policies."""
    run_evaluate(_load(config))


@app.command()
def report(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Write the result figures and the self-contained HTML dashboard."""
    run_report(_load(config))


@app.command(name="all")
def run_all(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run every stage in order."""
    cfg = _load(config)
    for stage in (run_data, run_analyze, run_train, run_evaluate, run_report):
        stage(cfg)
