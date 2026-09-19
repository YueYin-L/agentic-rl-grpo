from __future__ import annotations

import json

from research_router_opt.analysis_models import AnalysisTask, ModelReply, ToolCall
from research_router_opt.backend import MockBackend
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig


def _task(task_id: str = "runtime-task") -> AnalysisTask:
    return AnalysisTask(
        task_id=task_id,
        split="train",
        task_type="error_recovery",
        database_id="train_analytics",
        question="查询订单总数，遇到字段错误时恢复，并将结果除以 2。",
        expected_answer=360.0,
        reference_sql=("SELECT COUNT(*) AS value FROM orders",),
        expected_sql_results=(((720,),),),
        expected_tools=("schema", "sql", "calculator"),
        requires_error_recovery=True,
    )


def test_runtime_supports_multi_tool_error_recovery_and_serialization() -> None:
    backend = MockBackend(
        (
            ModelReply(tool_call=ToolCall("c1", "schema", {"table": "orders"})),
            ModelReply(tool_call=ToolCall("c2", "sql", {"query": "SELECT missing FROM orders"})),
            ModelReply(tool_call=ToolCall("c3", "schema", {"table": "orders"})),
            ModelReply(
                tool_call=ToolCall(
                    "c4", "sql", {"query": "SELECT COUNT(*) AS value FROM orders"}
                )
            ),
            ModelReply(
                tool_call=ToolCall(
                    "c5",
                    "calculator",
                    {"expression": "count / 2", "variables": {"count": 720}},
                )
            ),
            ModelReply(content="最终答案是 360。"),
        )
    )
    with StatefulToolEnvironment() as environment:
        trajectory = MultiTurnAgentRuntime(backend, environment).run(_task())
    assert trajectory.state.termination_reason == "final_answer"
    assert trajectory.state.step_count == 6
    assert [result.status for result in trajectory.state.tool_results] == [
        "ok",
        "error",
        "ok",
        "ok",
        "ok",
    ]
    assert "missing" in trajectory.state.errors[0]
    assistant_tool_message = trajectory.state.messages[2].to_dict()
    assert assistant_tool_message["tool_calls"][0]["function"]["name"] == "schema"
    assert assistant_tool_message["content"] is None
    json.dumps(trajectory.to_dict(), ensure_ascii=False)


def test_runtime_handles_malformed_call_max_steps_and_backend_replacement() -> None:
    malformed_backend = MockBackend((ModelReply(), ModelReply(content="无法完成")))
    with StatefulToolEnvironment() as environment:
        malformed = MultiTurnAgentRuntime(malformed_backend, environment).run(_task())
    assert malformed.state.tool_results[0].status == "invalid"
    assert malformed.state.termination_reason == "final_answer"

    looping_backend = MockBackend(
        (ModelReply(tool_call=ToolCall("loop", "schema", {})),), repeat_last=True
    )
    with StatefulToolEnvironment() as environment:
        looped = MultiTurnAgentRuntime(
            looping_backend,
            environment,
            RuntimeConfig(max_steps=2),
        ).run(_task())
    assert looped.state.termination_reason == "max_steps"
    assert looped.state.step_count == 2

    replacement_backend = MockBackend((ModelReply(content="replacement backend"),))
    with StatefulToolEnvironment() as environment:
        replaced = MultiTurnAgentRuntime(replacement_backend, environment).run(_task())
    assert replaced.backend_name == "mock"
    assert replaced.state.final_answer == "replacement backend"


def test_environment_reset_does_not_leak_previous_episode_state() -> None:
    backend = MockBackend(
        (
            ModelReply(
                tool_call=ToolCall(
                    "sql", "sql", {"query": "SELECT COUNT(*) AS value FROM orders"}
                )
            ),
            ModelReply(content="720"),
        )
    )
    with StatefulToolEnvironment() as environment:
        runtime = MultiTurnAgentRuntime(backend, environment)
        first = runtime.run(_task("first"))
        second = runtime.run(_task("second"))
        assert len(environment.calls) == 1
    assert first.state.task_id == "first"
    assert second.state.task_id == "second"
    assert len(second.state.tool_results) == 1
