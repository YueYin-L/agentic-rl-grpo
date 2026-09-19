"""Bounded multi-turn agent runtime independent of the model implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from research_router_opt.analysis_models import (
    AgentMessage,
    AgentState,
    AnalysisTask,
    AnalysisTrajectory,
    ToolResult,
)
from research_router_opt.backend import ModelBackend
from research_router_opt.environment import StatefulToolEnvironment

_SYSTEM_PROMPT = """You are a data-analysis agent. Inspect schema before relying on columns.
Use only the provided tools. SQL must be read-only. Treat tool errors as observations and recover
when possible. Return a final answer only after the result is grounded in tool output."""


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 10
    model_timeout_s: float = 60.0
    tool_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.model_timeout_s <= 0 or self.tool_timeout_s <= 0:
            raise ValueError("Timeouts must be positive.")


class MultiTurnAgentRuntime:
    def __init__(
        self,
        backend: ModelBackend,
        environment: StatefulToolEnvironment,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.backend = backend
        self.environment = environment
        self.config = config or RuntimeConfig()

    def run(self, task: AnalysisTask) -> AnalysisTrajectory:
        self.backend.reset()
        self.environment.reset(task)
        state = AgentState(
            task_id=task.task_id,
            messages=[
                AgentMessage(role="system", content=_SYSTEM_PROMPT),
                AgentMessage(role="user", content=task.question),
            ],
        )

        for _ in range(self.config.max_steps):
            state.step_count += 1
            try:
                reply = self.backend.generate(
                    state.messages,
                    self.environment.tool_specs,
                    timeout_s=self.config.model_timeout_s,
                )
            except TimeoutError:
                state.state = "failed"
                state.errors.append("Model generation timed out.")
                state.termination_reason = "model_timeout"
                break
            except Exception as exc:  # model SDK/network failures must become episode state
                state.state = "failed"
                state.errors.append(f"Model backend error: {exc}")
                state.termination_reason = "model_error"
                break

            if reply.tool_call is None and reply.content is not None:
                state.messages.append(AgentMessage(role="assistant", content=reply.content))
                state.final_answer = reply.content
                state.state = "completed"
                state.termination_reason = "final_answer"
                break

            if reply.tool_call is None:
                result = ToolResult(
                    call_id=f"malformed-{state.step_count}",
                    name="",
                    status="invalid",
                    error="Model reply contained neither a tool call nor a final answer.",
                )
                self._record_tool_result(state, result)
                continue

            call = reply.tool_call
            state.tool_calls.append(call)
            state.messages.append(
                AgentMessage(
                    role="assistant",
                    content=None,
                    tool_call=call,
                )
            )
            result = self.environment.step(call, timeout_s=self.config.tool_timeout_s)
            self._record_tool_result(state, result)
        else:
            state.state = "failed"
            state.termination_reason = "max_steps"
            state.errors.append(f"Agent exceeded max_steps={self.config.max_steps}.")

        return AnalysisTrajectory(task_id=task.task_id, backend_name=self.backend.name, state=state)

    @staticmethod
    def _record_tool_result(state: AgentState, result: ToolResult) -> None:
        state.tool_results.append(result)
        payload = result.to_dict()
        state.messages.append(
            AgentMessage(
                role="tool",
                name=result.name or "invalid_tool_call",
                tool_call_id=result.call_id,
                content=json.dumps(payload, ensure_ascii=False),
            )
        )
        if result.status != "ok":
            state.errors.append(result.error or f"{result.name} failed with {result.status}.")
