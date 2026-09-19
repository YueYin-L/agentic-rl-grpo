from __future__ import annotations

from research_router_opt.analysis_tasks import (
    DEFAULT_SPLIT_SIZES,
    TASK_TAXONOMY,
    build_analysis_tasks,
    load_analysis_tasks,
    write_analysis_datasets,
)


def test_task_generation_covers_taxonomy_and_keeps_splits_isolated(tmp_path) -> None:
    generated = {split: build_analysis_tasks(split) for split in DEFAULT_SPLIT_SIZES}
    assert {split: len(tasks) for split, tasks in generated.items()} == DEFAULT_SPLIT_SIZES
    for split, tasks in generated.items():
        assert {task.task_type for task in tasks} == set(TASK_TAXONOMY)
        assert all(task.split == split for task in tasks)
        assert len({task.task_id for task in tasks}) == len(tasks)
    all_ids = [{task.task_id for task in tasks} for tasks in generated.values()]
    assert all_ids[0].isdisjoint(all_ids[1])
    assert all_ids[0].isdisjoint(all_ids[2])
    assert all_ids[1].isdisjoint(all_ids[2])

    manifest = write_analysis_datasets(tmp_path)
    assert manifest["test"]["task_count"] == 150
    loaded = load_analysis_tasks(tmp_path / "analysis_test.jsonl")
    assert loaded == generated["test"]
