"""Typed models for the multi-turn data-analysis agent and its trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]
ToolStatus = Literal["ok", "error", "invalid", "timeout"]
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class AgentMessage:
    role: MessageRole
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_call: ToolCall | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_call is not None:
            payload["tool_calls"] = [
                {
                    "id": self.tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": self.tool_call.name,
                        "arguments": json.dumps(
                            self.tool_call.arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                }
            ]
        return payload


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    """One model turn: either a tool call or a final answer."""

    content: str | None = None
    tool_call: ToolCall | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    status: ToolStatus
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisTask:
    task_id: str
    split: SplitName
    task_type: str
    database_id: str
    question: str
    expected_answer: str | int | float
    reference_sql: tuple[str, ...]
    expected_sql_results: tuple[tuple[tuple[Any, ...], ...], ...]
    expected_tools: tuple[str, ...]
    requires_error_recovery: bool = False
    answer_tolerance: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    task_id: str
    messages: list[AgentMessage]
    state: str = "running"
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    step_count: int = 0
    final_answer: str | None = None
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisTrajectory:
    task_id: str
    backend_name: str
    state: AgentState

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "backend_name": self.backend_name,
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class VerificationResult:
    answer_correct: bool
    sql_exec_success: bool
    result_correct: bool
    tool_valid: bool
    schema_valid: bool
    grounded: bool
    recovered_from_error: bool
    meaningful_recovery: bool
    step_count: int
    redundant_calls: int
    failure_modes: tuple[str, ...]

    @property
    def success(self) -> bool:
        return (
            self.answer_correct
            and self.result_correct
            and self.tool_valid
            and self.grounded
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["success"] = self.success
        return payload
