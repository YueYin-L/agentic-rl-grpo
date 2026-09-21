"""CPU-safe configuration and evidence helpers for the CP6 GPU smoke test."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from research_router_opt.analysis_models import AnalysisTask


@dataclass(frozen=True)
class SmokeConfig:
    base_model: str
    revision: str
    reward_config: str
    dataset_path: str
    dataset_sha256: str
    train_task_ids: tuple[str, ...]
    group_size: int
    grpo_steps: int
    gradient_accumulation_sequences: int
    temperature: float
    top_p: float
    max_output_tokens: int
    max_steps: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    max_seq_length: int

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError("GRPO group_size must be greater than one.")
        if not self.train_task_ids:
            raise ValueError("At least one train task is required.")
        if self.grpo_steps < 1:
            raise ValueError("grpo_steps must be positive.")
        if self.gradient_accumulation_sequences < 1:
            raise ValueError("gradient_accumulation_sequences must be positive.")
        if self.temperature <= 0:
            raise ValueError("Smoke rollouts must use stochastic sampling.")


@dataclass(frozen=True)
class GroupAudit:
    task_id: str
    rewards: tuple[float, ...]
    mean: float
    std: float
    variance: float
    degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterMetadata:
    step: int
    path: str
    sha256: str
    file_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_smoke_config(path: Path) -> SmokeConfig:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = cast(dict[str, Any], payload["model"])
    reward = cast(dict[str, Any], payload["reward"])
    dataset = cast(dict[str, Any], payload["dataset"])
    rollout = cast(dict[str, Any], payload["rollout"])
    training = cast(dict[str, Any], payload["training"])
    lora = cast(dict[str, Any], payload["lora"])
    runtime = cast(dict[str, Any], payload["runtime"])
    return SmokeConfig(
        base_model=str(model["id"]),
        revision=str(model["revision"]),
        reward_config=str(reward["config"]),
        dataset_path=str(dataset["path"]),
        dataset_sha256=str(dataset["sha256"]),
        train_task_ids=tuple(str(value) for value in dataset["task_ids"]),
        group_size=int(rollout["group_size"]),
        grpo_steps=int(training["grpo_steps"]),
        gradient_accumulation_sequences=int(
            training["gradient_accumulation_sequences"]
        ),
        temperature=float(rollout["temperature"]),
        top_p=float(rollout["top_p"]),
        max_output_tokens=int(rollout["max_output_tokens"]),
        max_steps=int(runtime["max_steps"]),
        learning_rate=float(training["learning_rate"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        max_seq_length=int(model["max_seq_length"]),
    )


def select_train_tasks(
    tasks: list[AnalysisTask], config: SmokeConfig, dataset_bytes: bytes
) -> list[AnalysisTask]:
    actual_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if actual_hash != config.dataset_sha256:
        raise ValueError("Train dataset hash does not match the CP6 configuration.")
    by_id = {task.task_id: task for task in tasks}
    missing = sorted(set(config.train_task_ids) - set(by_id))
    if missing:
        raise ValueError(f"Configured train tasks are missing: {missing}")
    selected = [by_id[task_id] for task_id in config.train_task_ids]
    if any(task.split != "train" for task in selected):
        raise ValueError("CP6 may only use train-split tasks.")
    return selected


def audit_group(task_id: str, rewards: list[float]) -> GroupAudit:
    if len(rewards) <= 1:
        raise ValueError("A rollout group must contain multiple rewards.")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    return GroupAudit(
        task_id=task_id,
        rewards=tuple(rewards),
        mean=round(mean, 6),
        std=round(math.sqrt(variance), 6),
        variance=round(variance, 6),
        degenerate=math.isclose(variance, 0.0, abs_tol=1e-12),
    )


def require_non_degenerate_group(audits: list[GroupAudit]) -> None:
    if not audits or all(audit.degenerate for audit in audits):
        raise ValueError("All rollout groups are reward-degenerate; GRPO update is unsafe.")


def hash_adapter_directory(path: Path, *, step: int) -> AdapterMetadata:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Adapter directory contains no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return AdapterMetadata(
        step=step,
        path=str(path),
        sha256=digest.hexdigest(),
        file_count=len(files),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
