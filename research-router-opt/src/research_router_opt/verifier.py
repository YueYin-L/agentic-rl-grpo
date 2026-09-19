"""Deterministic programmatic verifier for complete agent trajectories."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from research_router_opt.analysis_models import (
    AnalysisTask,
    AnalysisTrajectory,
    ToolCall,
    ToolResult,
    VerificationResult,
)

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER.finditer(text.replace(",", ""))]


def _answer_correct(task: AnalysisTask, answer: str | None) -> bool:
    if answer is None:
        return False
    if isinstance(task.expected_answer, str):
        return task.expected_answer.casefold() in answer.casefold()
    expected = float(task.expected_answer)
    return any(
        math.isclose(value, expected, rel_tol=task.answer_tolerance, abs_tol=task.answer_tolerance)
        for value in _numbers(answer)
    )


def _value_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return str(left).strip().casefold() == str(right).strip().casefold()


def _rows_equal(
    actual: Iterable[Iterable[Any]],
    expected: Iterable[Iterable[Any]],
    tolerance: float,
) -> bool:
    actual_rows = [tuple(row) for row in actual]
    expected_rows = [tuple(row) for row in expected]
    if len(actual_rows) != len(expected_rows):
        return False
    unmatched = list(actual_rows)
    for expected_row in expected_rows:
        matched_index = next(
            (
                index
                for index, actual_row in enumerate(unmatched)
                if len(actual_row) == len(expected_row)
                and all(
                    _value_equal(actual_value, expected_value, tolerance)
                    for actual_value, expected_value in zip(
                        actual_row, expected_row, strict=True
                    )
                )
            ),
            None,
        )
        if matched_index is None:
            return False
        unmatched.pop(matched_index)
    return True


def _result_correct(task: AnalysisTask, results: list[ToolResult]) -> bool:
    actual_sql_results = [
        result.output.get("rows", [])
        for result in results
        if result.name == "sql" and result.status == "ok"
    ]
    return all(
        any(
            _rows_equal(actual, expected, task.answer_tolerance)
            for actual in actual_sql_results
        )
        for expected in task.expected_sql_results
    )


def _flatten_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_values(child)
    else:
        yield value


def _grounded(task: AnalysisTask, results: list[ToolResult], answer_correct: bool) -> bool:
    if not answer_correct:
        return False
    expected = task.expected_answer
    for result in results:
        if result.status != "ok":
            continue
        for value in _flatten_values(result.output):
            if _value_equal(value, expected, task.answer_tolerance):
                return True
    return False


def _recovered(results: list[ToolResult]) -> bool:
    failure_index = next(
        (index for index, result in enumerate(results) if result.status != "ok"),
        None,
    )
    if failure_index is None:
        return False
    return any(result.status == "ok" for result in results[failure_index + 1 :])


def _redundant_calls(calls: list[ToolCall]) -> int:
    signatures = [
        (call.name, json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
        for call in calls
    ]
    return sum(count - 1 for count in Counter(signatures).values() if count > 1)


def verify_trajectory(task: AnalysisTask, trajectory: AnalysisTrajectory) -> VerificationResult:
    state = trajectory.state
    answer_correct = _answer_correct(task, state.final_answer)
    sql_exec_success = any(
        result.name == "sql" and result.status == "ok" for result in state.tool_results
    )
    result_correct = _result_correct(task, state.tool_results)
    tool_valid = all(result.status != "invalid" for result in state.tool_results)
    schema_valid = (
        "schema" not in task.expected_tools
        or any(
            result.name == "schema" and result.status == "ok" for result in state.tool_results
        )
    )
    recovered = _recovered(state.tool_results)
    redundant_calls = _redundant_calls(state.tool_calls)
    grounded = _grounded(task, state.tool_results, answer_correct)
    failure_modes: list[str] = []
    if not answer_correct:
        failure_modes.append("wrong_answer")
    if not sql_exec_success:
        failure_modes.append("sql_execution_failure")
    if not result_correct:
        failure_modes.append("wrong_result")
    if not tool_valid:
        failure_modes.append("invalid_tool_call")
    if not schema_valid:
        failure_modes.append("schema_not_verified")
    if not grounded:
        failure_modes.append("ungrounded_answer")
    if task.requires_error_recovery and any(
        result.status != "ok" for result in state.tool_results
    ) and not recovered:
        failure_modes.append("recovery_failure")
    if state.termination_reason == "max_steps":
        failure_modes.append("loop_or_max_steps")
    if redundant_calls:
        failure_modes.append("redundant_tool_calls")
    return VerificationResult(
        answer_correct=answer_correct,
        sql_exec_success=sql_exec_success,
        result_correct=result_correct,
        tool_valid=tool_valid,
        schema_valid=schema_valid,
        grounded=grounded,
        recovered_from_error=recovered,
        step_count=state.step_count,
        redundant_calls=redundant_calls,
        failure_modes=tuple(failure_modes),
    )
