from pathlib import Path

from research_router_opt.data import build_datasets
from research_router_opt.policy import LinUCBPolicy
from research_router_opt.train import train_policy


def test_trained_policy_round_trips(tmp_path: Path) -> None:
    train_tasks, validation_tasks = build_datasets()
    policy = train_policy(train_tasks, epochs=4)
    artifact = tmp_path / "policy.json"
    policy.save(artifact)
    restored = LinUCBPolicy.load(artifact)
    for task in validation_tasks:
        assert restored.select_tool(task.query) == policy.select_tool(task.query)

