"""Typed domain models shared by training, evaluation, and the HTTP layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TOOL_NAMES = (
    "species_lookup",
    "paper_search",
    "evidence_compare",
    "abstract_summarize",
)


@dataclass(frozen=True)
class RoutingTask:
    task_id: str
    query: str
    expected_tool: str

    def __post_init__(self) -> None:
        if self.expected_tool not in TOOL_NAMES:
            raise ValueError(f"Unsupported expected tool: {self.expected_tool}")


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    status: str
    result: dict[str, Any]

    def has_valid_schema(self) -> bool:
        return self.tool in TOOL_NAMES and self.status in {"ok", "not_found"}


@dataclass(frozen=True)
class TrajectoryStep:
    observation: str
    action: str
    tool_output: ToolObservation
    latency_ms: float


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    policy_name: str
    steps: tuple[TrajectoryStep, ...]
    reward: float
    success: bool
    termination_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    policy_name: str
    task_count: int
    selection_accuracy: float
    average_reward: float
    invalid_action_rate: float
    average_tool_calls: float
    p50_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

