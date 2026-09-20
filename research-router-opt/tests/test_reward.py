import pytest

from research_router_opt.analysis_models import (
    AgentState,
    AnalysisTask,
    AnalysisTrajectory,
    ToolCall,
    ToolResult,
)
from research_router_opt.reward import RewardConfig, calculate_reward, score_agent_trajectory
from research_router_opt.tools import execute_tool
from research_router_opt.verifier import verify_trajectory


def test_correct_tool_receives_full_reward() -> None:
    observation = execute_tool("paper_search", "检索雪豹论文")
    reward = calculate_reward("paper_search", "paper_search", observation, 1)
    assert reward.total == pytest.approx(1.0)


def test_wrong_tool_is_penalized_but_schema_credit_is_visible() -> None:
    observation = execute_tool("species_lookup", "检索雪豹论文")
    reward = calculate_reward("species_lookup", "paper_search", observation, 1)
    assert reward.total == pytest.approx(-0.2)


def _task(task_type: str = "schema_discovery") -> AnalysisTask:
    return AnalysisTask(
        task_id="reward-task",
        split="validation",
        task_type=task_type,
        database_id="validation_analytics",
        question="return 2",
        expected_answer=2,
        reference_sql=("SELECT 2",),
        expected_sql_results=(((2,),),),
        expected_tools=("schema", "sql"),
    )


def _trajectory(
    calls: list[ToolCall],
    results: list[ToolResult],
    answer: str | None,
    *,
    termination: str = "final_answer",
) -> AnalysisTrajectory:
    return AnalysisTrajectory(
        task_id="reward-task",
        backend_name="mock",
        state=AgentState(
            task_id="reward-task",
            messages=[],
            state="completed" if answer is not None else "failed",
            tool_calls=calls,
            tool_results=results,
            errors=[result.error for result in results if result.error],
            step_count=len(calls) + (1 if answer is not None else 0),
            final_answer=answer,
            termination_reason=termination,
        ),
    )


def _score(task: AnalysisTask, trajectory: AnalysisTrajectory) -> float:
    verification = verify_trajectory(task, trajectory)
    return score_agent_trajectory(
        task,
        trajectory,
        verification,
        RewardConfig(),
    ).total_reward


def _correct_trajectory(*, termination: str = "final_answer") -> AnalysisTrajectory:
    calls = [
        ToolCall("s", "schema", {"table": "orders"}),
        ToolCall("q", "sql", {"query": "SELECT 2"}),
    ]
    results = [
        ToolResult("s", "schema", "ok", {"tables": {"orders": []}}),
        ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[2]]}),
    ]
    return _trajectory(calls, results, "2", termination=termination)


def test_agent_reward_success_exceeds_failure() -> None:
    task = _task()
    failure = _trajectory(
        [ToolCall("q", "sql", {"query": "SELECT 9"})],
        [ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[9]]})],
        "9",
    )
    assert _score(task, _correct_trajectory()) > _score(task, failure)


def test_agent_reward_grounded_correct_exceeds_unsupported_correct() -> None:
    task = _task()
    unsupported = _trajectory(
        [ToolCall("q", "sql", {"query": "SELECT 9"})],
        [ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[9]]})],
        "2",
    )
    assert _score(task, _correct_trajectory()) > _score(task, unsupported)


def test_agent_reward_efficient_success_exceeds_looping_success() -> None:
    task = _task()
    looping = _correct_trajectory(termination="max_steps")
    assert _score(task, _correct_trajectory()) > _score(task, looping)


def test_agent_reward_correct_arithmetic_exceeds_executable_wrong_semantics() -> None:
    task = _task("sql_calculator")
    correct = _correct_trajectory()
    wrong = _trajectory(
        [
            ToolCall("s", "schema", {"table": "orders"}),
            ToolCall("q", "sql", {"query": "SELECT 2"}),
            ToolCall("c", "calculator", {"expression": "2 * 2"}),
        ],
        [
            ToolResult("s", "schema", "ok", {"tables": {"orders": []}}),
            ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[2]]}),
            ToolResult("c", "calculator", "ok", {"value": 4}),
        ],
        "4",
    )
    assert _score(task, correct) > _score(task, wrong)


def test_agent_reward_meaningful_recovery_exceeds_superficial_recovery() -> None:
    task = _task()
    meaningful = _trajectory(
        [
            ToolCall("bad", "schema", {"table": "missing"}),
            ToolCall("s", "schema", {"table": "orders"}),
            ToolCall("q", "sql", {"query": "SELECT 2"}),
        ],
        [
            ToolResult("bad", "schema", "error", error="Unknown table: missing"),
            ToolResult("s", "schema", "ok", {"tables": {"orders": []}}),
            ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[2]]}),
        ],
        "2",
    )
    superficial = _trajectory(
        [
            ToolCall("bad", "schema", {"table": "missing"}),
            ToolCall("q", "sql", {"query": "SELECT 2"}),
        ],
        [
            ToolResult("bad", "schema", "error", error="Unknown table: missing"),
            ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[2]]}),
        ],
        "2",
    )
    assert _score(task, meaningful) > _score(task, superficial)


def test_agent_reward_tool_spam_does_not_increase_reward() -> None:
    task = _task()
    efficient = _correct_trajectory()
    spam_calls = [ToolCall(f"s{i}", "schema", {"table": "orders"}) for i in range(6)]
    spam_results = [
        ToolResult(f"s{i}", "schema", "ok", {"tables": {"orders": []}})
        for i in range(6)
    ]
    spam_calls.append(ToolCall("q", "sql", {"query": "SELECT 2"}))
    spam_results.append(
        ToolResult("q", "sql", "ok", {"columns": ["value"], "rows": [[2]]})
    )
    spam = _trajectory(spam_calls, spam_results, "2")
    assert _score(task, efficient) > _score(task, spam)
