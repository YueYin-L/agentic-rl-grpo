"""Transparent reward calculation for tool-routing trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from research_router_opt.models import ToolObservation


@dataclass(frozen=True)
class RewardBreakdown:
    tool_selection: float
    task_success: float
    schema_validity: float
    call_cost: float

    @property
    def total(self) -> float:
        return round(
            self.tool_selection + self.task_success + self.schema_validity + self.call_cost,
            6,
        )


def calculate_reward(
    selected_tool: str,
    expected_tool: str,
    observation: ToolObservation,
    tool_calls: int,
) -> RewardBreakdown:
    if tool_calls < 1:
        raise ValueError("tool_calls must be at least 1")

    correct_tool = selected_tool == expected_tool
    return RewardBreakdown(
        tool_selection=0.65 if correct_tool else -0.35,
        task_success=0.20 if correct_tool and observation.status == "ok" else 0.0,
        schema_validity=0.20 if observation.has_valid_schema() else 0.0,
        call_cost=-0.05 * tool_calls,
    )

