"""Evaluation and report serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

from research_router_opt.agent import ResearchRouterAgent
from research_router_opt.models import EvaluationMetrics, RoutingTask, Trajectory
from research_router_opt.policy import RoutingPolicy


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(math.ceil(percentile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def evaluate_policy(
    policy: RoutingPolicy, tasks: list[RoutingTask]
) -> tuple[EvaluationMetrics, list[Trajectory]]:
    if not tasks:
        raise ValueError("At least one task is required for evaluation.")
    agent = ResearchRouterAgent(policy)
    trajectories = [agent.run(task) for task in tasks]
    latencies = [step.latency_ms for trajectory in trajectories for step in trajectory.steps]
    invalid_actions = sum(
        trajectory.termination_reason == "invalid_action" for trajectory in trajectories
    )
    metrics = EvaluationMetrics(
        policy_name=policy.name,
        task_count=len(tasks),
        selection_accuracy=sum(trajectory.success for trajectory in trajectories) / len(tasks),
        average_reward=sum(trajectory.reward for trajectory in trajectories) / len(tasks),
        invalid_action_rate=invalid_actions / len(tasks),
        average_tool_calls=sum(len(trajectory.steps) for trajectory in trajectories) / len(tasks),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )
    return metrics, trajectories


def write_evaluation(
    report_path: Path,
    metrics: EvaluationMetrics,
    trajectories: list[Trajectory],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "metrics": metrics.to_dict(),
                "trajectories": [trajectory.to_dict() for trajectory in trajectories],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

