from __future__ import annotations

import pytest

from research_router_opt.final_evaluation import (
    exact_mcnemar_p_value,
    metric_deltas,
    paired_transitions,
)


def _record(task_id: str, task_type: str, success: bool) -> dict[str, object]:
    return {
        "task": {"task_id": task_id, "task_type": task_type},
        "verification": {"success": success},
        "trajectory": {"state": {"tool_calls": []}},
        "failure_modes": [] if success else ["grounding_failure"],
    }


def test_paired_transitions_counts_and_groups() -> None:
    base = [_record("a", "schema", True), _record("b", "multi", False)]
    grpo = [_record("a", "schema", False), _record("b", "multi", True)]
    result = paired_transitions(base, grpo)
    assert result["overall"]["base_success_grpo_fail"] == 1
    assert result["overall"]["base_fail_grpo_success"] == 1
    assert result["by_task_type"]["schema"]["base_success_grpo_fail"] == 1


def test_paired_transitions_rejects_mismatched_tasks() -> None:
    with pytest.raises(ValueError, match="task IDs"):
        paired_transitions([_record("a", "schema", True)], [_record("b", "schema", True)])


def test_exact_mcnemar() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(0, 6) == pytest.approx(0.03125)
    assert exact_mcnemar_p_value(3, 3) == 1.0


def test_metric_deltas() -> None:
    names = (
        "task_success",
        "final_answer_accuracy",
        "sql_execution_success",
        "result_correctness",
        "tool_validity",
        "invalid_call_rate",
        "grounded_answer_rate",
        "recovery_rate",
        "average_steps",
        "average_tool_calls",
        "max_step_termination_rate",
        "reward_mean",
    )
    base = {name: 0.5 for name in names}
    grpo = {name: 0.75 for name in names}
    result = metric_deltas(base, grpo)
    assert result["task_success"]["absolute_delta"] == pytest.approx(0.25)
    assert result["task_success"]["relative_change"] == pytest.approx(0.5)
