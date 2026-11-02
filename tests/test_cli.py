"""End to end: every CLI stage on the fake dataset, then determinism of the outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from persuadables.cli import app
from tests.fakes import write_tiny_toml

RESULTS = (
    "data_validation.json",
    "readout.json",
    "training.json",
    "uplift_test.json",
    "policy.json",
)
FIGURES = (
    "eda_outcomes.png",
    "eda_spend.png",
    "eda_balance.png",
    "eda_recency.png",
    "effects.png",
    "segments.png",
    "qini.png",
    "learners.png",
    "calibration.png",
    "shap.png",
    "policy.png",
)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory, fake_source: Path) -> Path:
    workdir = tmp_path_factory.mktemp("cli")
    config = write_tiny_toml(workdir / "tiny.toml", fake_source)
    runner = CliRunner()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)
        result = runner.invoke(app, ["all", "--config", str(config)])
    assert result.exit_code == 0, result.output
    return workdir


def test_every_stage_writes_its_outputs(pipeline_run: Path) -> None:
    for name in RESULTS:
        assert (pipeline_run / "results" / name).is_file(), name
    for name in FIGURES:
        assert (pipeline_run / "docs" / "assets" / name).stat().st_size > 10_000, name
    assert (pipeline_run / "artifacts" / "predictions.npz").is_file()


def test_results_have_the_expected_shape(pipeline_run: Path) -> None:
    results = pipeline_run / "results"
    readout = json.loads((results / "readout.json").read_text(encoding="utf-8"))
    assert len(readout["effects"]) == 6
    assert {e["metric"] for e in readout["effects"]} == {"visit", "conversion", "spend"}
    policy = json.loads((results / "policy.json").read_text(encoding="utf-8"))
    names = {row["policy"] for row in policy["table"]}
    assert {"no_one", "everyone_arm1", "everyone_arm2", "uplift_multi", "random"} <= names
    training = json.loads((results / "training.json").read_text(encoding="utf-8"))
    assert set(training["selected"]) == {"visit", "conversion"}
    counts = training["outcome_counts"]
    assert sum(arm["customers"] for split in counts.values() for arm in split.values()) == 6000
    assert sum(arm["customers"] for arm in counts["test"].values()) == training["n_test"]
    power = readout["effects"][0]["power"]
    assert power["reference_relative_lift"] == pytest.approx(0.1)
    assert power["n_per_arm_for_reference_lift"] > 0
    assert 0 < power["power_at_reference_lift"] < 1
    assert 0 < readout["spend_zero_share"] < 1


def test_dashboard_embeds_the_results(pipeline_run: Path) -> None:
    html = (pipeline_run / "reports" / "dashboard.html").read_text(encoding="utf-8")
    assert "/*PAYLOAD*/" not in html
    assert "<!--PLOTLY-->" not in html
    start = html.index("const DATA = ") + len("const DATA = ")
    payload = json.loads(html[start : html.index(";\n", start)].replace("<\\/", "</"))
    assert payload["meta"]["customers"] == 6000
    assert payload["meta"]["checksPassed"] == payload["meta"]["checks"]


def test_rerunning_a_stage_reproduces_its_results(pipeline_run: Path) -> None:
    results = pipeline_run / "results"
    figures = pipeline_run / "docs" / "assets"
    before = {results / name: (results / name).read_bytes() for name in RESULTS}
    before |= {figures / name: (figures / name).read_bytes() for name in FIGURES}
    runner = CliRunner()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(pipeline_run)
        # `data` rebuilds the database from the raw file and rewrites data_validation.json.
        for stage in ("data", "analyze", "train", "evaluate", "report"):
            outcome = runner.invoke(app, [stage, "--config", "tiny.toml"])
            assert outcome.exit_code == 0, outcome.output
    for path, content in before.items():
        assert path.read_bytes() == content, f"{path.name} changed on rerun"


def test_missing_inputs_fail_with_a_hint(tmp_path: Path, fake_source: Path) -> None:
    config = write_tiny_toml(tmp_path / "tiny.toml", fake_source)
    runner = CliRunner()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        result = runner.invoke(app, ["analyze", "--config", str(config)])
    assert result.exit_code != 0
    assert "persuadables data" in str(result.exception)
