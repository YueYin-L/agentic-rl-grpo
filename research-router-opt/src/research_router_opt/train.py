"""Offline training loop for the contextual-bandit router."""

from __future__ import annotations

import random

from research_router_opt.agent import ResearchRouterAgent
from research_router_opt.models import RoutingTask
from research_router_opt.policy import LinUCBPolicy


def train_policy(
    tasks: list[RoutingTask],
    *,
    epochs: int = 12,
    alpha: float = 0.8,
    seed: int = 20260918,
) -> LinUCBPolicy:
    if not tasks:
        raise ValueError("At least one training task is required.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    policy = LinUCBPolicy(alpha=alpha)
    agent = ResearchRouterAgent(policy)
    randomizer = random.Random(seed)
    ordered_tasks = list(tasks)
    for _ in range(epochs):
        randomizer.shuffle(ordered_tasks)
        for task in ordered_tasks:
            trajectory = agent.run(task)
            policy.update(task.query, trajectory.steps[0].action, trajectory.reward)
    return policy

