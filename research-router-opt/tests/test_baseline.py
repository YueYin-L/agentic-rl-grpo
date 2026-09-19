from __future__ import annotations

import json

from research_router_opt.analysis_models import AnalysisTask, ModelReply, ToolCall
from research_router_opt.backend import MockBackend
from research_router_opt.baseline import run_baseline


def test_baseline_persists_trajectory_and_metrics(tmp_path) -> None:
    task = AnalysisTask(
        task_id="baseline-task",
        split="validation",
        task_type="schema_discovery",
        database_id="validation_analytics",
        question="统计订单数",
        expected_answer=720,
        reference_sql=("SELECT COUNT(*) AS value FROM orders",),
        expected_sql_results=(((720,),),),
        expected_tools=("schema", "sql"),
    )
    backend = MockBackend(
        (
            ModelReply(tool_call=ToolCall("s", "schema", {"table": "orders"})),
            ModelReply(
                tool_call=ToolCall(
                    "q", "sql", {"query": "SELECT COUNT(*) AS value FROM orders"}
                )
            ),
            ModelReply(content="订单数为 720。"),
        )
    )
    output = tmp_path / "baseline-run"
    summary = run_baseline(tasks=[task], backend=backend, output_dir=output)
    assert summary["overall"]["success"] == 1.0
    assert (output / "metrics.json").exists()
    trajectory_lines = (output / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trajectory_lines) == 1
    assert json.loads(trajectory_lines[0])["verification"]["success"] is True
