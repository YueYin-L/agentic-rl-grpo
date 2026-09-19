import pytest

from research_router_opt.reward import calculate_reward
from research_router_opt.tools import execute_tool


def test_correct_tool_receives_full_reward() -> None:
    observation = execute_tool("paper_search", "检索雪豹论文")
    reward = calculate_reward("paper_search", "paper_search", observation, 1)
    assert reward.total == pytest.approx(1.0)


def test_wrong_tool_is_penalized_but_schema_credit_is_visible() -> None:
    observation = execute_tool("species_lookup", "检索雪豹论文")
    reward = calculate_reward("species_lookup", "paper_search", observation, 1)
    assert reward.total == pytest.approx(-0.2)

