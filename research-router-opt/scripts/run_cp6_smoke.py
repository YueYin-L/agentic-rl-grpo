"""Run CP6 ART GRPO smoke training or fresh-process adapter reload on Linux GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from research_router_opt.analysis_models import AnalysisTask, AnalysisTrajectory
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.backend import VLLMBackend
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.grpo_smoke import (
    audit_group,
    hash_adapter_directory,
    load_smoke_config,
    require_non_degenerate_group,
    select_train_tasks,
    write_json,
)
from research_router_opt.reward import load_reward_config, score_agent_trajectory
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig
from research_router_opt.verifier import verify_trajectory


class GpuMonitor:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                values = [int(value.strip()) for value in completed.stdout.splitlines()]
                self.peak_mib = max([self.peak_mib, *values])
            self._stop.wait(1.0)

    def __enter__(self) -> GpuMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def _full_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _runtime_config(payload: dict[str, Any]) -> RuntimeConfig:
    values = cast(dict[str, Any], payload["runtime"])
    return RuntimeConfig(
        max_steps=int(values["max_steps"]),
        model_timeout_s=float(values["model_timeout_s"]),
        tool_timeout_s=float(values["tool_timeout_s"]),
    )


def _art_model(config: Any, payload: dict[str, Any]) -> Any:
    import art

    lora = cast(dict[str, Any], payload["lora"])
    return art.TrainableModel(
        name="qwen35-9b-cp6-smoke",
        run_name="cp6-minimal-grpo-smoke",
        project="agentic-rl-grpo",
        base_model=config.base_model,
        lora_config=art.LoRAConfig(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            max_seq_length=config.max_seq_length,
            use_gradient_checkpointing="unsloth",
            random_state=20260920,
        ),
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=config.max_seq_length,
                dtype="bfloat16",
                load_in_4bit=False,
                load_in_8bit=False,
                load_in_16bit=True,
                revision=config.revision,
                fast_inference=True,
                gpu_memory_utilization=0.62,
                max_lora_rank=config.lora_rank,
                random_state=20260920,
            ),
            trainer_args=art.dev.TrainerArgs(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                bf16=True,
                max_grad_norm=0.1,
                logging_steps=1,
            ),
            chat_template_tool_schema_format="vllm_openai",
        ),
    )


def _choice_history(
    trajectory: AnalysisTrajectory, raw_choices: list[dict[str, Any]]
) -> list[Any]:
    from openai.types.chat.chat_completion import Choice

    items: list[Any] = []
    choice_index = 0
    for message in trajectory.state.messages:
        if message.role == "assistant":
            if choice_index >= len(raw_choices):
                raise RuntimeError("Captured ART choices do not match assistant turns.")
            items.append(Choice.model_validate(raw_choices[choice_index]))
            choice_index += 1
        else:
            items.append(message.to_dict())
    if choice_index != len(raw_choices):
        raise RuntimeError("Unused captured choices remain after trajectory conversion.")
    return items


async def _rollout(
    *,
    model: Any,
    task: AnalysisTask,
    config: Any,
    payload: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import art

    reward_config = load_reward_config(Path(config.reward_config))
    backend = VLLMBackend(
        base_url=str(model.inference_base_url),
        model=model.get_inference_name(),
        api_key=str(model.inference_api_key or "EMPTY"),
        max_tokens=config.max_output_tokens,
        seed=seed,
        temperature=config.temperature,
        top_p=config.top_p,
        capture_choices=True,
    )
    with StatefulToolEnvironment() as environment:
        runtime = MultiTurnAgentRuntime(backend, environment, _runtime_config(payload))
        started = time.perf_counter()
        trajectory = await asyncio.to_thread(runtime.run, task)
        elapsed_s = time.perf_counter() - started
        tool_specs = list(environment.tool_specs)
    verification = verify_trajectory(task, trajectory)
    reward = score_agent_trajectory(task, trajectory, verification, reward_config)
    art_trajectory = art.Trajectory(
        messages_and_choices=_choice_history(trajectory, backend.raw_choices),
        tools=tool_specs,
        reward=reward.total_reward,
        metrics={
            "success": verification.success,
            "steps": verification.step_count,
            "tool_calls": len(trajectory.state.tool_calls),
        },
        metadata={"task_id": task.task_id, "split": task.split, "seed": seed},
    )
    record = {
        "task": task.to_dict(),
        "trajectory": trajectory.to_dict(),
        "verification": verification.to_dict(),
        "reward": reward.to_dict(),
        "elapsed_s": elapsed_s,
        "art_choice_count": len(backend.raw_choices),
    }
    return art_trajectory, record


def _tensor_delta(before_path: Path, after_path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    before = load_file(str(before_path / "adapter_model.safetensors"), device="cpu")
    after = load_file(str(after_path / "adapter_model.safetensors"), device="cpu")
    common = sorted(set(before) & set(after))
    if not common:
        raise RuntimeError("No common LoRA tensors found for parameter-change evidence.")
    squared = 0.0
    changed = 0
    max_abs = 0.0
    for name in common:
        delta = (after[name].float() - before[name].float()).cpu()
        norm = float(delta.norm().item())
        if norm > 0:
            changed += 1
        squared += norm * norm
        max_abs = max(max_abs, float(delta.abs().max().item()))
    return {
        "common_tensor_count": len(common),
        "changed_tensor_count": changed,
        "global_delta_l2": squared**0.5,
        "max_abs_delta": max_abs,
    }


async def train(config_path: Path, output_dir: Path) -> None:
    import art
    import torch
    from art.local.backend import LocalBackend
    from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

    config = load_smoke_config(config_path)
    payload = _full_config(config_path)
    dataset_path = Path(config.dataset_path)
    tasks = select_train_tasks(
        load_analysis_tasks(dataset_path), config, dataset_path.read_bytes()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = LocalBackend(path=str(cast(dict[str, Any], payload["art"])["path"]))
    model = _art_model(config, payload)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with GpuMonitor() as gpu:
        await model.register(backend)
        registered_step = await model.get_step()
        if registered_step != 0:
            raise RuntimeError(f"CP6 smoke must start at step 0, found {registered_step}.")
        groups: list[Any] = []
        records: list[dict[str, Any]] = []
        audits = []
        rollout_started = time.perf_counter()
        for task_index, task in enumerate(tasks):
            group_trajectories = []
            group_rewards = []
            for rollout_index in range(config.group_size):
                trajectory, record = await _rollout(
                    model=model,
                    task=task,
                    config=config,
                    payload=payload,
                    seed=20260920 + task_index * 100 + rollout_index,
                )
                group_trajectories.append(trajectory)
                group_rewards.append(float(trajectory.reward))
                records.append(record)
            groups.append(art.TrajectoryGroup(group_trajectories))
            audits.append(audit_group(task.task_id, group_rewards))
        rollout_time_s = time.perf_counter() - rollout_started
        require_non_degenerate_group(audits)
        model_dir = Path(get_model_dir(model, str(cast(dict[str, Any], payload["art"])["path"])))
        step0 = Path(get_step_checkpoint_dir(str(model_dir), 0))
        before = hash_adapter_directory(step0, step=0)
        training_started = time.perf_counter()
        result = await backend.train(
            model,
            groups,
            learning_rate=config.learning_rate,
            loss_fn=str(cast(dict[str, Any], payload["training"])["loss_fn"]),
            scale_rewards=True,
            logprob_calculation_chunk_size=64,
            packed_sequence_length=config.max_seq_length,
            grad_accumulation_sequences=4,
            save_checkpoint=True,
            verbose=True,
        )
        training_time_s = time.perf_counter() - training_started
        step1 = Path(get_step_checkpoint_dir(str(model_dir), int(result.step)))
        after = hash_adapter_directory(step1, step=int(result.step))
        delta = _tensor_delta(step0, step1)
        total_time_s = time.perf_counter() - started

    with (output_dir / "pre_update_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    evidence = {
        "phase": "train",
        "config": payload,
        "train_task_ids": [task.task_id for task in tasks],
        "group_audits": [audit.to_dict() for audit in audits],
        "training": {
            "initial_step": registered_step,
            "result_step": int(result.step),
            "metrics": dict(result.metrics),
            "rollout_time_s": rollout_time_s,
            "training_time_s": training_time_s,
            "total_time_s": total_time_s,
        },
        "adapter_before": before.to_dict(),
        "adapter_after": after.to_dict(),
        "parameter_delta": delta,
        "compute": {
            "peak_gpu_memory_mib_nvidia_smi": gpu.peak_mib,
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "oom": False,
        },
    }
    write_json(output_dir / "train_evidence.json", evidence)


async def reload(config_path: Path, output_dir: Path) -> None:
    from art.local.backend import LocalBackend

    config = load_smoke_config(config_path)
    payload = _full_config(config_path)
    dataset_path = Path(config.dataset_path)
    tasks = select_train_tasks(
        load_analysis_tasks(dataset_path), config, dataset_path.read_bytes()
    )
    backend = LocalBackend(path=str(cast(dict[str, Any], payload["art"])["path"]))
    model = _art_model(config, payload)
    started = time.perf_counter()
    with GpuMonitor() as gpu:
        await model.register(backend)
        step = await model.get_step()
        if step < 1:
            raise RuntimeError(f"Fresh reload expected trained step >= 1, found {step}.")
        trajectory, record = await _rollout(
            model=model,
            task=tasks[0],
            config=config,
            payload=payload,
            seed=20269999,
        )
        elapsed_s = time.perf_counter() - started
    write_json(
        output_dir / "reload_evidence.json",
        {
            "phase": "fresh_process_reload",
            "loaded_step": step,
            "reward": trajectory.reward,
            "post_update_record": record,
            "elapsed_s": elapsed_s,
            "peak_gpu_memory_mib_nvidia_smi": gpu.peak_mib,
            "process_boundary": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "reload"))
    parser.add_argument("--config", type=Path, default=Path("configs/cp6_grpo_smoke.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "train":
        asyncio.run(train(args.config, args.output))
    else:
        asyncio.run(reload(args.config, args.output))


if __name__ == "__main__":
    main()
