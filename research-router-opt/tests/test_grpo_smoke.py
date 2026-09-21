from dataclasses import replace
from pathlib import Path

import pytest

from research_router_opt.analysis_models import AnalysisTask
from research_router_opt.grpo_smoke import (
    audit_group,
    hash_adapter_directory,
    load_smoke_config,
    require_non_degenerate_group,
    select_train_tasks,
)


def _task(split: str = "train") -> AnalysisTask:
    return AnalysisTask(
        task_id=f"{split}-task",
        split=split,  # type: ignore[arg-type]
        task_type="sql_calculator",
        database_id=f"{split}_analytics",
        question="question",
        expected_answer=1,
        reference_sql=("SELECT 1",),
        expected_sql_results=(((1,),),),
        expected_tools=("schema", "sql", "calculator"),
    )


def test_smoke_config_loads_multiple_rollouts() -> None:
    config = load_smoke_config(Path("configs/cp6_grpo_smoke.toml"))
    assert config.group_size == 4
    assert config.grpo_steps == 1
    assert config.gradient_accumulation_sequences == 1
    assert config.trainer_num_generations == 1
    assert config.reward_config == "configs/cp5_reward.toml"


def test_group_audit_detects_variation_and_degeneracy() -> None:
    varied = audit_group("train-task", [-2.8, 5.5, -2.8, 5.5])
    same = audit_group("train-task", [5.5, 5.5, 5.5, 5.5])
    assert varied.variance > 0
    assert not varied.degenerate
    assert same.degenerate
    require_non_degenerate_group([same, varied])
    with pytest.raises(ValueError, match="reward-degenerate"):
        require_non_degenerate_group([same])


def test_train_selection_rejects_non_train_split() -> None:
    config = load_smoke_config(Path("configs/cp6_grpo_smoke.toml"))
    payload = b"frozen"
    import hashlib

    config = replace(
        config,
        dataset_sha256=hashlib.sha256(payload).hexdigest(),
        train_task_ids=("validation-task",),
    )
    with pytest.raises(ValueError, match="only use train"):
        select_train_tasks([_task("validation")], config, payload)


def test_adapter_metadata_uses_content_hash(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"before")
    before = hash_adapter_directory(adapter, step=0)
    (adapter / "adapter_model.safetensors").write_bytes(b"after")
    after = hash_adapter_directory(adapter, step=1)
    assert before.sha256 != after.sha256
    assert after.file_count == 1
