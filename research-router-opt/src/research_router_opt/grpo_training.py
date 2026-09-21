"""CPU-safe configuration and monitoring helpers for CP7 formal GRPO training."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class FormalTrainingConfig:
    base_model: str
    revision: str
    precision: str
    quantization: str
    max_seq_length: int
    reward_config: str
    reward_sha256: str
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    test_path: str
    test_sha256: str
    group_size: int
    temperature: float
    top_p: float
    max_output_tokens: int
    max_steps: int
    model_timeout_s: float
    tool_timeout_s: float
    learning_rate: float
    max_update_steps: int
    gradient_accumulation_sequences: int
    trainer_num_generations: int
    logprob_calculation_chunk_size: int
    validate_every: int
    save_every: int
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float

    def __post_init__(self) -> None:
        if self.group_size != 4:
            raise ValueError("CP7 frozen group_size must be 4.")
        if self.learning_rate != 5e-6:
            raise ValueError("CP7 frozen learning rate must be 5e-6.")
        if self.max_update_steps != 100:
            raise ValueError("CP7 frozen update budget must be 100 steps.")
        if self.validate_every != 20 or self.save_every != 20:
            raise ValueError("CP7 validation and checkpoint cadence must be 20 steps.")
        if self.lora_rank != 8 or self.lora_alpha != 16 or self.lora_dropout != 0:
            raise ValueError("CP7 frozen LoRA configuration does not match CP6.")
        if self.max_seq_length != 4096:
            raise ValueError("CP7 frozen context length must be 4096.")


def load_formal_training_config(path: Path) -> FormalTrainingConfig:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    model = cast(dict[str, Any], payload["model"])
    reward = cast(dict[str, Any], payload["reward"])
    dataset = cast(dict[str, Any], payload["dataset"])
    runtime = cast(dict[str, Any], payload["runtime"])
    rollout = cast(dict[str, Any], payload["rollout"])
    training = cast(dict[str, Any], payload["training"])
    checkpoint = cast(dict[str, Any], payload["checkpoint"])
    validation = cast(dict[str, Any], payload["validation"])
    lora = cast(dict[str, Any], payload["lora"])
    return FormalTrainingConfig(
        base_model=str(model["id"]),
        revision=str(model["revision"]),
        precision=str(model["precision"]),
        quantization=str(model["quantization"]),
        max_seq_length=int(model["max_seq_length"]),
        reward_config=str(reward["config"]),
        reward_sha256=str(reward["sha256"]),
        train_path=str(dataset["train_path"]),
        train_sha256=str(dataset["train_sha256"]),
        validation_path=str(dataset["validation_path"]),
        validation_sha256=str(dataset["validation_sha256"]),
        test_path=str(dataset["test_path"]),
        test_sha256=str(dataset["test_sha256"]),
        group_size=int(rollout["group_size"]),
        temperature=float(rollout["temperature"]),
        top_p=float(rollout["top_p"]),
        max_output_tokens=int(rollout["max_output_tokens"]),
        max_steps=int(runtime["max_steps"]),
        model_timeout_s=float(runtime["model_timeout_s"]),
        tool_timeout_s=float(runtime["tool_timeout_s"]),
        learning_rate=float(training["learning_rate"]),
        max_update_steps=int(training["max_update_steps"]),
        gradient_accumulation_sequences=int(
            training["gradient_accumulation_sequences"]
        ),
        trainer_num_generations=int(training["trainer_num_generations"]),
        logprob_calculation_chunk_size=int(
            training["logprob_calculation_chunk_size"]
        ),
        validate_every=int(validation["every_steps"]),
        save_every=int(checkpoint["save_every_steps"]),
        seed=int(training["seed"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_hashes(config: FormalTrainingConfig, root: Path) -> dict[str, str]:
    expected = {
        "reward": (config.reward_config, config.reward_sha256),
        "train": (config.train_path, config.train_sha256),
        "validation": (config.validation_path, config.validation_sha256),
        "test": (config.test_path, config.test_sha256),
    }
    actual: dict[str, str] = {}
    for name, (relative_path, expected_hash) in expected.items():
        digest = sha256_file(root / relative_path)
        if digest != expected_hash:
            raise ValueError(f"Frozen {name} hash mismatch: {digest} != {expected_hash}")
        actual[name] = digest
    return actual


def checkpoint_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["task_success"]),
        float(metrics["grounded_answer_rate"]),
        float(metrics["result_correctness"]),
        float(metrics["final_answer_accuracy"]),
        -float(metrics["invalid_call_rate"]),
        -float(metrics["average_steps"]),
    )


def stop_reasons(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if previous is None:
        return []
    reasons: list[str] = []
    reward_up = float(current["reward_mean"]) > float(previous["reward_mean"])
    success_down = float(current["task_success"]) < float(previous["task_success"])
    if reward_up and success_down:
        reasons.append("validation_reward_up_but_task_success_down")
    if (
        float(previous["grounded_answer_rate"])
        - float(current["grounded_answer_rate"])
        >= 0.05
    ):
        reasons.append("grounding_decreased_by_at_least_5_points")
    if (
        float(current["invalid_call_rate"])
        - float(previous["invalid_call_rate"])
        >= 0.03
    ):
        reasons.append("invalid_call_rate_increased_by_at_least_3_points")
    if (
        float(current["max_step_termination_rate"])
        - float(previous["max_step_termination_rate"])
        >= 0.05
    ):
        reasons.append("max_step_termination_increased_by_at_least_5_points")
    return reasons
