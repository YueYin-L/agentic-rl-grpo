"""Bounded one-step research router agent."""

from __future__ import annotations

from time import perf_counter

from research_router_opt.models import RoutingTask, Trajectory, TrajectoryStep
from research_router_opt.policy import RoutingPolicy
from research_router_opt.reward import calculate_reward
from research_router_opt.tools import execute_tool


class ResearchRouterAgent:
    def __init__(self, policy: RoutingPolicy) -> None:
        self.policy = policy

    def run(self, task: RoutingTask) -> Trajectory:
        selected_tool = self.policy.select_tool(task.query)
        started = perf_counter()
        observation = execute_tool(selected_tool, task.query)
        latency_ms = (perf_counter() - started) * 1000.0
        breakdown = calculate_reward(
            selected_tool=selected_tool,
            expected_tool=task.expected_tool,
            observation=observation,
            tool_calls=1,
        )
        success = selected_tool == task.expected_tool and observation.status == "ok"
        step = TrajectoryStep(
            observation=task.query,
            action=selected_tool,
            tool_output=observation,
            latency_ms=latency_ms,
        )
        return Trajectory(
            task_id=task.task_id,
            policy_name=self.policy.name,
            steps=(step,),
            reward=breakdown.total,
            success=success,
            termination_reason="task_completed" if success else "wrong_tool",
        )

