"""Small reproducible task set with disjoint training and validation templates."""

from __future__ import annotations

import json
from pathlib import Path

from research_router_opt.models import RoutingTask

_TRAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "species_lookup": (
        "查询{subject}的学名和栖息地",
        "给出{subject}的保护级别与分布区域",
        "我在论文里看到{subject}，先查这个物种的基础信息",
    ),
    "paper_search": (
        "检索关于{subject}识别方法的最新论文",
        "搜索{subject}相关文献和发表年份",
        "帮我查找研究{subject}种群变化的论文",
    ),
    "evidence_compare": (
        "比较两篇{subject}论文的方法差异和实验结论",
        "对比多个{subject}研究的证据，哪个结论更可靠",
        "分析{subject}相关文献之间的优缺点",
    ),
    "abstract_summarize": (
        "总结这篇{subject}论文的摘要",
        "概括已有{subject}文献的研究结论",
        "提炼这段{subject}研究摘要的方法与结果",
    ),
}

_VAL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "species_lookup": (
        "不要搜索论文，直接告诉我{subject}的物种分布",
        "比较资料之前，先查询{subject}的学名",
    ),
    "paper_search": (
        "不用总结，搜索发表过的{subject}研究文献",
        "为了比较证据，请先检索{subject}方向的论文",
    ),
    "evidence_compare": (
        "不要只做摘要，请比较两篇{subject}论文的实验方法",
        "已有多篇{subject}文献，分析它们的结论差异",
    ),
    "abstract_summarize": (
        "不用再检索，请总结现有{subject}论文摘要",
        "读完多篇{subject}文献后，归纳摘要中的主要结论",
    ),
}

_TRAIN_SUBJECTS = ("雪豹", "大熊猫", "金丝猴", "哺乳动物图像分类")
_VAL_SUBJECTS = ("亚洲象", "穿山甲", "长臂猿")


def _expand(
    templates: dict[str, tuple[str, ...]], subjects: tuple[str, ...], split: str
) -> list[RoutingTask]:
    tasks: list[RoutingTask] = []
    for tool, tool_templates in templates.items():
        for template_index, template in enumerate(tool_templates):
            for subject_index, subject in enumerate(subjects):
                task_id = f"{split}-{tool}-{template_index:02d}-{subject_index:02d}"
                tasks.append(
                    RoutingTask(
                        task_id=task_id,
                        query=template.format(subject=subject),
                        expected_tool=tool,
                    )
                )
    return tasks


def build_datasets() -> tuple[list[RoutingTask], list[RoutingTask]]:
    train = _expand(_TRAIN_TEMPLATES, _TRAIN_SUBJECTS, "train")
    validation = _expand(_VAL_TEMPLATES, _VAL_SUBJECTS, "val")
    return train, validation


def write_datasets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation = build_datasets()
    train_path = output_dir / "train_tasks.jsonl"
    validation_path = output_dir / "val_tasks.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    return train_path, validation_path


def _write_jsonl(path: Path, tasks: list[RoutingTask]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "query": task.query,
                        "expected_tool": task.expected_tool,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

