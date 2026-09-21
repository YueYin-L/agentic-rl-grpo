from pathlib import Path

import pytest

from research_router_opt.grpo_training import (
    checkpoint_rank,
    load_formal_training_config,
    stop_reasons,
    verify_frozen_hashes,
)


def test_cp7_frozen_configuration() -> None:
    config = load_formal_training_config(Path("configs/cp7_grpo_formal.toml"))
    assert config.base_model == "Qwen/Qwen3.5-9B"
    assert config.group_size == 4
    assert config.max_update_steps == 100
    assert config.validate_every == 20
    assert config.save_every == 20
    assert config.lora_rank == 8
    assert config.max_seq_length == 4096
    hashes = verify_frozen_hashes(config, Path.cwd())
    assert set(hashes) == {"reward", "train", "validation", "test"}


def test_checkpoint_rank_prefers_success_then_grounding() -> None:
    base = {
        "task_success": 0.5,
        "grounded_answer_rate": 0.5,
        "result_correctness": 0.5,
        "final_answer_accuracy": 0.5,
        "invalid_call_rate": 0.0,
        "average_steps": 5.0,
    }
    better = dict(base, task_success=0.51, average_steps=9.0)
    assert checkpoint_rank(better) > checkpoint_rank(base)


def test_stop_rules_detect_reward_hacking_and_behavior_regression() -> None:
    previous = {
        "reward_mean": 1.0,
        "task_success": 0.5,
        "grounded_answer_rate": 0.5,
        "invalid_call_rate": 0.0,
        "max_step_termination_rate": 0.1,
    }
    current = {
        "reward_mean": 1.1,
        "task_success": 0.49,
        "grounded_answer_rate": 0.44,
        "invalid_call_rate": 0.04,
        "max_step_termination_rate": 0.16,
    }
    reasons = stop_reasons(previous, current)
    assert "validation_reward_up_but_task_success_down" in reasons
    assert "grounding_decreased_by_at_least_5_points" in reasons
    assert "invalid_call_rate_increased_by_at_least_3_points" in reasons
    assert "max_step_termination_increased_by_at_least_5_points" in reasons


def test_frozen_config_rejects_budget_change(tmp_path: Path) -> None:
    source = Path("configs/cp7_grpo_formal.toml").read_text(encoding="utf-8")
    path = tmp_path / "changed.toml"
    path.write_text(source.replace("max_update_steps = 100", "max_update_steps = 99"))
    with pytest.raises(ValueError, match="100 steps"):
        load_formal_training_config(path)
