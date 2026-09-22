"""Pure helpers for the paired CP8 frozen-test comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def paired_transitions(
    base_records: list[dict[str, Any]], grpo_records: list[dict[str, Any]]
) -> dict[str, Any]:
    base = {record["task"]["task_id"]: record for record in base_records}
    grpo = {record["task"]["task_id"]: record for record in grpo_records}
    if set(base) != set(grpo):
        raise ValueError("Base and GRPO task IDs do not match.")
    counts = {
        "both_success": 0,
        "base_success_grpo_fail": 0,
        "base_fail_grpo_success": 0,
        "both_fail": 0,
    }
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in counts})
    tasks: list[dict[str, Any]] = []
    for task_id in sorted(base):
        base_record = base[task_id]
        grpo_record = grpo[task_id]
        base_ok = bool(base_record["verification"]["success"])
        grpo_ok = bool(grpo_record["verification"]["success"])
        if base_ok and grpo_ok:
            transition = "both_success"
        elif base_ok:
            transition = "base_success_grpo_fail"
        elif grpo_ok:
            transition = "base_fail_grpo_success"
        else:
            transition = "both_fail"
        task_type = str(base_record["task"]["task_type"])
        counts[transition] += 1
        by_type[task_type][transition] += 1
        tasks.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "transition": transition,
                "base_failure_modes": base_record.get("failure_modes", []),
                "grpo_failure_modes": grpo_record.get("failure_modes", []),
                "base_tool_calls": len(base_record["trajectory"]["state"]["tool_calls"]),
                "grpo_tool_calls": len(grpo_record["trajectory"]["state"]["tool_calls"]),
            }
        )
    return {"overall": counts, "by_task_type": dict(sorted(by_type.items())), "tasks": tasks}


def exact_mcnemar_p_value(base_only: int, grpo_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    total = base_only + grpo_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(base_only, grpo_only) + 1))
    return float(min(1.0, 2.0 * tail / (2**total)))


def metric_deltas(base: dict[str, Any], grpo: dict[str, Any]) -> dict[str, Any]:
    metrics = (
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
    result: dict[str, Any] = {}
    for name in metrics:
        base_value = float(base[name])
        grpo_value = float(grpo[name])
        delta = grpo_value - base_value
        entry: dict[str, float | None] = {
            "base": base_value,
            "grpo": grpo_value,
            "absolute_delta": delta,
        }
        entry["relative_change"] = delta / base_value if base_value else None
        result[name] = entry
    return result
