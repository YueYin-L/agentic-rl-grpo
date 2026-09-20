from __future__ import annotations

from research_router_opt.analysis_models import (
    AgentState,
    AnalysisTask,
    AnalysisTrajectory,
    ToolCall,
    ToolResult,
)
from research_router_opt.verifier import verify_trajectory


def _task(*, recovery: bool = False) -> AnalysisTask:
    return AnalysisTask(
        task_id="verify-task",
        split="validation",
        task_type="error_recovery" if recovery else "schema_discovery",
        database_id="validation_analytics",
        question="统计订单数",
        expected_answer=2,
        reference_sql=("SELECT COUNT(*) FROM orders",),
        expected_sql_results=(((2,),),),
        expected_tools=("schema", "sql"),
        requires_error_recovery=recovery,
    )


def _trajectory(
    *,
    calls: list[ToolCall],
    results: list[ToolResult],
    answer: str | None,
    termination: str = "final_answer",
) -> AnalysisTrajectory:
    state = AgentState(
        task_id="verify-task",
        messages=[],
        state="completed" if answer is not None else "failed",
        tool_calls=calls,
        tool_results=results,
        step_count=len(calls) + (1 if answer is not None else 0),
        final_answer=answer,
        termination_reason=termination,
    )
    return AnalysisTrajectory(task_id="verify-task", backend_name="mock", state=state)


def test_verifier_accepts_semantically_equal_result_from_different_sql() -> None:
    calls = [
        ToolCall("s", "schema", {}),
        ToolCall("q", "sql", {"query": "SELECT SUM(1) FROM (VALUES (1), (1))"}),
    ]
    results = [
        ToolResult("s", "schema", "ok", {"tables": {"orders": []}}),
        ToolResult("q", "sql", "ok", {"columns": ["count"], "rows": [[2]]}),
    ]
    verified = verify_trajectory(_task(), _trajectory(calls=calls, results=results, answer="2"))
    assert verified.success is True
    assert verified.result_correct is True
    assert verified.grounded is True


def test_verifier_accepts_scalar_result_derived_from_detail_rows() -> None:
    calls = [
        ToolCall("q", "sql", {"query": "SELECT order_id FROM orders"}),
        ToolCall("c", "calculator", {"expression": "1 + 1"}),
    ]
    results = [
        ToolResult(
            "q",
            "sql",
            "ok",
            {"columns": ["order_id"], "rows": [[10], [11]], "row_count": 2},
        ),
        ToolResult("c", "calculator", "ok", {"value": 2}),
    ]
    verified = verify_trajectory(_task(), _trajectory(calls=calls, results=results, answer="2"))
    assert verified.result_correct is True
    assert verified.grounded is True


def test_verifier_accepts_scalar_result_grounded_in_sql_row_count() -> None:
    calls = [ToolCall("q", "sql", {"query": "SELECT order_id FROM orders"})]
    results = [
        ToolResult(
            "q",
            "sql",
            "ok",
            {"columns": ["order_id"], "rows": [[10], [11]], "row_count": 2},
        )
    ]
    verified = verify_trajectory(_task(), _trajectory(calls=calls, results=results, answer="2"))
    assert verified.result_correct is True
    assert verified.grounded is True


def test_verifier_accepts_final_calculator_outcome_without_reference_intermediates() -> None:
    task = AnalysisTask(
        task_id="multi-query",
        split="validation",
        task_type="multi_query_aggregation",
        database_id="validation_analytics",
        question="比较两个区域",
        expected_answer=5.0,
        reference_sql=("SELECT 0.15", "SELECT 0.10"),
        expected_sql_results=(((0.15,),), ((0.10,),)),
        expected_tools=("schema", "sql", "sql", "calculator"),
    )
    calls = [
        ToolCall("q1", "sql", {"query": "SELECT 3, 20"}),
        ToolCall("q2", "sql", {"query": "SELECT 1, 10"}),
        ToolCall("c", "calculator", {"expression": "(3/20 - 1/10) * 100"}),
    ]
    results = [
        ToolResult("q1", "sql", "ok", {"columns": ["refunds", "orders"], "rows": [[3, 20]]}),
        ToolResult("q2", "sql", "ok", {"columns": ["refunds", "orders"], "rows": [[1, 10]]}),
        ToolResult("c", "calculator", "ok", {"value": 5.0}),
    ]
    verified = verify_trajectory(task, _trajectory(calls=calls, results=results, answer="5.0"))
    assert verified.result_correct is True
    assert verified.grounded is True
    assert verified.success is True


def test_verifier_detects_wrong_answer_malformed_repeat_and_grounding_failure() -> None:
    repeated = ToolCall("q1", "sql", {"query": "SELECT wrong FROM orders"})
    calls = [repeated, ToolCall("q2", "sql", repeated.arguments), ToolCall("x", "missing", {})]
    results = [
        ToolResult("q1", "sql", "error", error="wrong column"),
        ToolResult("q2", "sql", "error", error="wrong column"),
        ToolResult("x", "missing", "invalid", error="unknown tool"),
    ]
    verified = verify_trajectory(
        _task(recovery=True),
        _trajectory(calls=calls, results=results, answer="答案是 99"),
    )
    assert verified.success is False
    assert verified.redundant_calls == 1
    assert set(verified.failure_modes) >= {
        "wrong_answer",
        "sql_execution_failure",
        "wrong_result",
        "invalid_tool_call",
        "schema_not_verified",
        "ungrounded_answer",
        "recovery_failure",
        "redundant_tool_calls",
    }


def test_verifier_recognizes_error_recovery_and_hallucinated_answer() -> None:
    calls = [
        ToolCall("bad", "sql", {"query": "SELECT area FROM customers"}),
        ToolCall("schema", "schema", {"table": "customers"}),
        ToolCall("good", "sql", {"query": "SELECT 2"}),
    ]
    results = [
        ToolResult("bad", "sql", "error", error="wrong column"),
        ToolResult("schema", "schema", "ok", {"tables": {"customers": []}}),
        ToolResult("good", "sql", "ok", {"columns": ["value"], "rows": [[2]]}),
    ]
    recovered = verify_trajectory(
        _task(recovery=True), _trajectory(calls=calls, results=results, answer="2")
    )
    hallucinated = verify_trajectory(
        _task(recovery=True), _trajectory(calls=calls, results=results, answer="3")
    )
    assert recovered.recovered_from_error is True
    assert recovered.meaningful_recovery is True
    assert recovered.success is True
    assert hallucinated.answer_correct is False
    assert hallucinated.grounded is False
    assert hallucinated.meaningful_recovery is False


def test_verifier_rejects_superficial_recovery() -> None:
    calls = [
        ToolCall("bad", "schema", {"table": "missing"}),
        ToolCall("unrelated", "schema", {"table": "orders"}),
    ]
    results = [
        ToolResult("bad", "schema", "error", error="Unknown table: missing"),
        ToolResult("unrelated", "schema", "ok", {"tables": {"orders": []}}),
    ]
    verified = verify_trajectory(
        _task(recovery=True),
        _trajectory(calls=calls, results=results, answer="2"),
    )
    assert verified.recovered_from_error is True
    assert verified.meaningful_recovery is False
