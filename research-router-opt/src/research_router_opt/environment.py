"""Stateful tool-use environment with episode reset and trajectory-safe observations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from research_router_opt.analysis_models import AnalysisTask, ToolCall, ToolResult
from research_router_opt.analysis_tools import AnalysisToolRegistry, create_analysis_database

RegistryFactory = Callable[[str], AnalysisToolRegistry]


class StatefulToolEnvironment:
    def __init__(self, registry_factory: RegistryFactory | None = None) -> None:
        self._registry_factory = registry_factory or self._default_registry_factory
        self._registry: AnalysisToolRegistry | None = None
        self.current_task: AnalysisTask | None = None
        self.calls: list[ToolCall] = []
        self.results: list[ToolResult] = []

    @staticmethod
    def _default_registry_factory(database_id: str) -> AnalysisToolRegistry:
        return AnalysisToolRegistry(create_analysis_database(database_id))

    @property
    def tool_specs(self) -> Sequence[dict[str, Any]]:
        if self._registry is None:
            raise RuntimeError("Environment must be reset before tool specs are requested.")
        return self._registry.tool_specs

    def reset(self, task: AnalysisTask) -> None:
        self.close()
        self.current_task = task
        self.calls = []
        self.results = []
        self._registry = self._registry_factory(task.database_id)

    def step(self, call: ToolCall, *, timeout_s: float) -> ToolResult:
        if self._registry is None or self.current_task is None:
            raise RuntimeError("Environment must be reset before executing a tool.")
        self.calls.append(call)
        result = self._registry.execute(call, timeout_s=timeout_s)
        self.results.append(result)
        return result

    def close(self) -> None:
        if self._registry is not None:
            self._registry.close()
        self._registry = None

    def __enter__(self) -> StatefulToolEnvironment:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
