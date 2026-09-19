from pathlib import Path

from research_router_opt.cli import run_pipeline


def test_pipeline_writes_reproducible_artifacts(tmp_path: Path) -> None:
    summary = run_pipeline(tmp_path)
    assert summary["training_tasks"] == 48
    assert summary["validation_tasks"] == 24
    assert (tmp_path / "artifacts" / "policy.json").exists()
    assert (tmp_path / "reports" / "comparison.md").exists()
    assert (tmp_path / "reports" / "failure_analysis.md").exists()
    baseline = summary["baseline"]
    optimized = summary["optimized"]
    assert isinstance(baseline, dict)
    assert isinstance(optimized, dict)
    assert optimized["selection_accuracy"] > baseline["selection_accuracy"]
