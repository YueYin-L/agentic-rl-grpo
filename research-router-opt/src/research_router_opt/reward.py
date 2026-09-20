"""Transparent reward calculation for tool-routing trajectories."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_router_opt.analysis_models import (
    AnalysisTask,
    AnalysisTrajectory,
    VerificationResult,
)
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


@dataclass(frozen=True)
class RewardConfig:
    """Interpretable trajectory reward with outcome-dominant positive mass."""

    answer_correct: float = 2.0
    result_correct: float = 2.0
    grounded: float = 1.5
    meaningful_recovery: float = 0.4
    multi_observation: float = 0.4
    wrong_answer_penalty: float = 1.0
    wrong_result_penalty: float = 1.0
    ungrounded_penalty: float = 0.8
    tool_error_penalty: float = 0.15
    redundant_call_penalty: float = 0.2
    repeated_nonexistent_probe_penalty: float = 0.25
    excess_call_penalty: float = 0.15
    max_steps_penalty: float = 1.0
    max_counted_tool_errors: int = 3
    max_counted_redundant_calls: int = 3
    max_counted_nonexistent_probes: int = 3
    max_counted_excess_calls: int = 3


@dataclass(frozen=True)
class AgentRewardBreakdown:
    answer_reward: float
    result_reward: float
    grounding_reward: float
    recovery_reward: float
    multi_observation_reward: float
    wrong_answer_penalty: float
    wrong_result_penalty: float
    ungrounded_penalty: float
    tool_error_penalty: float
    redundant_call_penalty: float
    nonexistent_probe_penalty: float
    excess_call_penalty: float
    max_steps_penalty: float
    tool_error_count: int
    redundant_call_count: int
    repeated_nonexistent_probe_count: int
    excess_call_count: int

    @property
    def total_reward(self) -> float:
        values = self.outcome_components | self.process_components | self.penalties
        return round(sum(values.values()), 6)

    @property
    def outcome_components(self) -> dict[str, float]:
        return {
            "answer_reward": self.answer_reward,
            "result_reward": self.result_reward,
            "grounding_reward": self.grounding_reward,
        }

    @property
    def process_components(self) -> dict[str, float]:
        return {
            "recovery_reward": self.recovery_reward,
            "multi_observation_reward": self.multi_observation_reward,
        }

    @property
    def penalties(self) -> dict[str, float]:
        return {
            "wrong_answer_penalty": self.wrong_answer_penalty,
            "wrong_result_penalty": self.wrong_result_penalty,
            "ungrounded_penalty": self.ungrounded_penalty,
            "tool_error_penalty": self.tool_error_penalty,
            "redundant_call_penalty": self.redundant_call_penalty,
            "nonexistent_probe_penalty": self.nonexistent_probe_penalty,
            "excess_call_penalty": self.excess_call_penalty,
            "max_steps_penalty": self.max_steps_penalty,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "outcome_components": self.outcome_components,
            "process_components": self.process_components,
            "penalties": self.penalties,
            "counts": {
                "tool_errors": self.tool_error_count,
                "redundant_calls": self.redundant_call_count,
                "repeated_nonexistent_probes": self.repeated_nonexistent_probe_count,
                "excess_calls": self.excess_call_count,
            },
        }


_MISSING_OBJECT = re.compile(
    r"(?:unknown table:\s*|table with name\s+)([\w.-]+)",
    flags=re.IGNORECASE,
)


def _multi_observation_completed(
    task: AnalysisTask,
    trajectory: AnalysisTrajectory,
    verification: VerificationResult,
) -> bool:
    if task.task_type != "multi_query_aggregation":
        return False
    successful_tools = [
        result.name for result in trajectory.state.tool_results if result.status == "ok"
    ]
    return (
        successful_tools.count("sql") >= 2
        and "calculator" in successful_tools
        and verification.result_correct
        and verification.grounded
    )


def _repeated_nonexistent_probes(trajectory: AnalysisTrajectory) -> int:
    missing_objects = [
        match.group(1).casefold()
        for result in trajectory.state.tool_results
        if result.status != "ok" and result.error
        for match in _MISSING_OBJECT.finditer(result.error)
    ]
    return max(0, len(missing_objects) - 1)


def score_agent_trajectory(
    task: AnalysisTask,
    trajectory: AnalysisTrajectory,
    verification: VerificationResult,
    config: RewardConfig,
) -> AgentRewardBreakdown:
    """Score one complete trajectory without rewarding saturated execution signals."""
    state = trajectory.state
    tool_error_count = sum(result.status != "ok" for result in state.tool_results)
    repeated_nonexistent = _repeated_nonexistent_probes(trajectory)
    excess_calls = max(0, len(state.tool_calls) - (len(task.expected_tools) + 2))
    multi_observation = _multi_observation_completed(task, trajectory, verification)
    return AgentRewardBreakdown(
        answer_reward=config.answer_correct if verification.answer_correct else 0.0,
        result_reward=config.result_correct if verification.result_correct else 0.0,
        grounding_reward=config.grounded if verification.grounded else 0.0,
        recovery_reward=(
            config.meaningful_recovery if verification.meaningful_recovery else 0.0
        ),
        multi_observation_reward=config.multi_observation if multi_observation else 0.0,
        wrong_answer_penalty=(
            -config.wrong_answer_penalty if not verification.answer_correct else 0.0
        ),
        wrong_result_penalty=(
            -config.wrong_result_penalty if not verification.result_correct else 0.0
        ),
        ungrounded_penalty=-config.ungrounded_penalty if not verification.grounded else 0.0,
        tool_error_penalty=-config.tool_error_penalty
        * min(tool_error_count, config.max_counted_tool_errors),
        redundant_call_penalty=-config.redundant_call_penalty
        * min(verification.redundant_calls, config.max_counted_redundant_calls),
        nonexistent_probe_penalty=-config.repeated_nonexistent_probe_penalty
        * min(repeated_nonexistent, config.max_counted_nonexistent_probes),
        excess_call_penalty=-config.excess_call_penalty
        * min(excess_calls, config.max_counted_excess_calls),
        max_steps_penalty=(
            -config.max_steps_penalty if state.termination_reason == "max_steps" else 0.0
        ),
        tool_error_count=tool_error_count,
        redundant_call_count=verification.redundant_calls,
        repeated_nonexistent_probe_count=repeated_nonexistent,
        excess_call_count=excess_calls,
    )


def load_reward_config(path: Path) -> RewardConfig:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    values = payload.get("reward", payload)
    if not isinstance(values, dict):
        raise ValueError("Reward configuration must be a TOML table.")
    return RewardConfig(**values)
