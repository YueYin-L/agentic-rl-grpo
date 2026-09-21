"""Run the frozen CP7 formal GRPO experiment on the validated Linux GPU stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import time
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from run_cp6_smoke import GpuMonitor, _art_model, _openai_server_config, _rollout

from research_router_opt.analysis_models import AnalysisTask
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.backend import VLLMBackend
from research_router_opt.baseline import (
    EvaluatedTrajectory,
    _aggregate,
    _classify_failures,
)
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.grpo_smoke import audit_group, require_non_degenerate_group, write_json
from research_router_opt.grpo_training import (
    checkpoint_rank,
    load_formal_training_config,
    stop_reasons,
    verify_frozen_hashes,
)
from research_router_opt.reward import load_reward_config, score_agent_trajectory
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig
from research_router_opt.verifier import verify_trajectory


def _full_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(f"Tracked working tree is not clean:\n{status}")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "tracked_working_tree_clean": True,
    }


def _runtime_config(config: Any) -> RuntimeConfig:
    return RuntimeConfig(
        max_steps=config.max_steps,
        model_timeout_s=config.model_timeout_s,
        tool_timeout_s=config.tool_timeout_s,
    )


def _validation_metrics(
    *,
    model: Any,
    tasks: list[AnalysisTask],
    config: Any,
    payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validation = cast(dict[str, Any], payload["validation"])
    reward_config = load_reward_config(Path(config.reward_config))
    backend = VLLMBackend(
        base_url=str(model.inference_base_url),
        model=model.get_inference_name(),
        api_key=str(model.inference_api_key or "EMPTY"),
        max_tokens=int(validation["max_output_tokens"]),
        seed=int(validation["seed"]),
        temperature=float(validation["temperature"]),
        top_p=float(validation["top_p"]),
    )
    records: list[EvaluatedTrajectory] = []
    grouped: dict[str, list[EvaluatedTrajectory]] = defaultdict(list)
    rewards: list[float] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "trajectories.jsonl"
    with path.open("w", encoding="utf-8") as handle, StatefulToolEnvironment() as env:
        runtime = MultiTurnAgentRuntime(backend, env, _runtime_config(config))
        for task in tasks:
            started = time.perf_counter()
            trajectory = runtime.run(task)
            elapsed_s = time.perf_counter() - started
            verification = verify_trajectory(task, trajectory)
            reward = score_agent_trajectory(
                task, trajectory, verification, reward_config
            )
            record = EvaluatedTrajectory(task, trajectory, verification, elapsed_s)
            records.append(record)
            grouped[task.task_type].append(record)
            rewards.append(reward.total_reward)
            handle.write(
                json.dumps(
                    {
                        "task": task.to_dict(),
                        "trajectory": trajectory.to_dict(),
                        "verification": verification.to_dict(),
                        "reward": reward.to_dict(),
                        "failure_modes": sorted(_classify_failures(record)),
                        "elapsed_s": elapsed_s,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            handle.flush()
    overall = _aggregate(records)
    overall["reward_mean"] = statistics.mean(rewards)
    overall["reward_std"] = statistics.pstdev(rewards)
    overall["max_step_termination_rate"] = sum(
        record.trajectory.state.termination_reason == "max_steps" for record in records
    ) / len(records)
    metrics = {
        "overall": overall,
        "by_task_type": {
            task_type: _aggregate(task_records)
            for task_type, task_records in sorted(grouped.items())
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    state: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    validations = cast(list[dict[str, Any]], state["validations"])
    lines = [
        "# CP7 Formal GRPO Training Report",
        "",
        "## Status",
        "",
        f"- status: {state['status']}",
        f"- completed update steps: {state['current_step']}",
        f"- best validation step: {state.get('best_step')}",
        f"- stop reasons: {', '.join(state.get('stop_reasons', [])) or 'none'}",
        "",
        "No frozen-test task was loaded or evaluated during CP7.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Training Timeline",
        "",
        "| Step | Task | Reward mean | Reward std | Rollout s | Train s |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in cast(list[dict[str, Any]], state["updates"]):
        lines.append(
            f"| {entry['step']} | {entry['task_id']} | {entry['reward_mean']:.4f} | "
            f"{entry['reward_std']:.4f} | {entry['rollout_time_s']:.1f} | "
            f"{entry['training_time_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint List",
            "",
        ]
    )
    for checkpoint in cast(list[dict[str, Any]], state["checkpoints"]):
        tags = []
        if checkpoint["step"] == state.get("best_step"):
            tags.append("best-validation")
        if checkpoint["step"] == state["current_step"]:
            tags.append("last")
        lines.append(
            f"- step {checkpoint['step']}: `{checkpoint['path']}` "
            f"({', '.join(tags) or 'milestone'})"
        )
    lines.extend(
        [
            "",
            "## Validation Metrics by Checkpoint",
            "",
            "| Step | Success | Final acc. | Result | Grounded | Recovery | Avg steps | "
            "Avg tools | Max-step | Reward mean/std |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for entry in validations:
        metrics = entry["overall"]
        lines.append(
            f"| {entry['step']} | {metrics['task_success']:.4f} | "
            f"{metrics['final_answer_accuracy']:.4f} | {metrics['result_correctness']:.4f} | "
            f"{metrics['grounded_answer_rate']:.4f} | {metrics['recovery_rate']:.4f} | "
            f"{metrics['average_steps']:.2f} | {metrics['average_tool_calls']:.2f} | "
            f"{metrics['max_step_termination_rate']:.4f} | "
            f"{metrics['reward_mean']:.4f}/{metrics['reward_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Checkpoint Selection",
            "",
            "Checkpoints are ranked by Task Success, then grounded rate, result correctness, "
            "final accuracy, lower invalid-call rate, and lower average steps.",
            "",
            f"Selected step: {state.get('best_step')}",
            "",
            "## Failure Analysis",
            "",
        ]
    )
    if validations:
        best = next(
            entry for entry in validations if entry["step"] == state.get("best_step")
        )
        failures = cast(dict[str, int], best["overall"]["failure_modes"])
        lines.extend(f"- {name}: {count}" for name, count in failures.items())
    else:
        lines.append("- No validation checkpoint completed yet.")
    lines.extend(
        [
            "",
            "## Comparison against CP4 Baseline",
            "",
            f"- CP4 Task Success: {float(baseline['task_success']):.4f}",
            f"- CP4 Final Accuracy: {float(baseline['final_answer_accuracy']):.4f}",
            f"- CP4 Result Correctness: {float(baseline['result_correctness']):.4f}",
            f"- CP4 Grounded Rate: {float(baseline['grounded_answer_rate']):.4f}",
            "",
        ]
    )
    if validations:
        best_metrics = next(
            entry["overall"]
            for entry in validations
            if entry["step"] == state.get("best_step")
        )
        delta = float(best_metrics["task_success"]) - float(baseline["task_success"])
        lines.append(f"Best validation Task Success delta: {delta:+.4f}")
    else:
        lines.append("No improvement claim is possible before validation completes.")
    lines.extend(
        [
            "",
            "CP7 does not establish held-out improvement. CP8 frozen-test evaluation requires "
            "separate review and authorization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


async def train(config_path: Path, output_dir: Path) -> None:
    import art
    from art.local.backend import LocalBackend
    from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

    project_root = Path.cwd()
    config = load_formal_training_config(config_path)
    payload = _full_config(config_path)
    frozen_hashes = verify_frozen_hashes(config, project_root)
    git = _git_info(project_root)
    train_tasks = load_analysis_tasks(Path(config.train_path))
    validation_tasks = load_analysis_tasks(Path(config.validation_path))
    if len(train_tasks) != 300 or len(validation_tasks) != 150:
        raise RuntimeError("Frozen CP7 split sizes must be train=300 and validation=150.")
    if any(task.split != "train" for task in train_tasks):
        raise RuntimeError("CP7 training input contains a non-train task.")
    if any(task.split != "validation" for task in validation_tasks):
        raise RuntimeError("CP7 validation input contains a non-validation task.")

    output_dir.mkdir(parents=True, exist_ok=False)
    art_config = cast(dict[str, Any], payload["art"])
    backend = LocalBackend(path=str(art_config["path"]))
    model = _art_model(
        config,
        payload,
        model_name=str(art_config["model_name"]),
        run_name=str(art_config["run_name"]),
        project=str(art_config["project"]),
    )
    baseline = cast(dict[str, Any], payload["baseline"])
    manifest = {
        "run_id": output_dir.name,
        "started_at": datetime.now(UTC).isoformat(),
        "git": git,
        "frozen_hashes": frozen_hashes,
        "config": payload,
        "train_task_count": len(train_tasks),
        "validation_task_count": len(validation_tasks),
        "test_loaded": False,
        "flashinfer_sampler_disabled": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
        == "0",
    }
    write_json(output_dir / "manifest.json", manifest)
    state: dict[str, Any] = {
        "status": "running",
        "current_step": 0,
        "group_attempts": 0,
        "updates": [],
        "validations": [],
        "checkpoints": [],
        "best_step": None,
        "stop_reasons": [],
    }
    write_json(output_dir / "training_state.json", state)

    schedule = list(train_tasks)
    random.Random(config.seed).shuffle(schedule)
    training_values = cast(dict[str, Any], payload["training"])
    max_group_attempts = int(training_values["max_group_attempts"])
    training_log = output_dir / "training_trajectories.jsonl"
    model_dir: Path | None = None
    best_metrics: dict[str, Any] | None = None
    with GpuMonitor() as gpu:
        await model.register(
            backend,
            _openai_client_config=_openai_server_config(art, payload),
        )
        initial_step = int(await model.get_step())
        if initial_step != 0:
            raise RuntimeError(f"CP7 must start from base step 0, found {initial_step}.")
        model_dir = Path(get_model_dir(model, str(art_config["path"])))
        with training_log.open("a", encoding="utf-8") as trajectory_handle:
            while int(state["current_step"]) < config.max_update_steps:
                if int(state["group_attempts"]) >= max_group_attempts:
                    raise RuntimeError("Exceeded bounded group attempts before 100 updates.")
                task_index = int(state["group_attempts"]) % len(schedule)
                task = schedule[task_index]
                state["group_attempts"] = int(state["group_attempts"]) + 1
                target_step = int(state["current_step"]) + 1
                rollout_started = time.perf_counter()
                trajectories = []
                records = []
                rewards = []
                for rollout_index in range(config.group_size):
                    trajectory, record = await _rollout(
                        model=model,
                        task=task,
                        config=config,
                        payload=payload,
                        seed=config.seed
                        + int(state["group_attempts"]) * 100
                        + rollout_index,
                    )
                    trajectories.append(trajectory)
                    records.append(record)
                    rewards.append(float(trajectory.reward))
                rollout_time_s = time.perf_counter() - rollout_started
                audit = audit_group(task.task_id, rewards)
                for record in records:
                    trajectory_handle.write(
                        json.dumps(
                            {
                                "target_step": target_step,
                                "group_attempt": state["group_attempts"],
                                **record,
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                trajectory_handle.flush()
                if audit.degenerate:
                    print(
                        json.dumps(
                            {
                                "event": "degenerate_group_skipped",
                                "target_step": target_step,
                                "task_id": task.task_id,
                                "rewards": rewards,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    write_json(output_dir / "training_state.json", state)
                    continue
                require_non_degenerate_group([audit])
                save_checkpoint = target_step % config.save_every == 0
                training_started = time.perf_counter()
                result = await backend.train(
                    model,
                    [art.TrajectoryGroup(trajectories)],
                    learning_rate=config.learning_rate,
                    loss_fn=str(training_values["loss_fn"]),
                    scale_rewards=bool(training_values["scale_rewards"]),
                    logprob_calculation_chunk_size=config.logprob_calculation_chunk_size,
                    packed_sequence_length=config.max_seq_length,
                    grad_accumulation_sequences=config.gradient_accumulation_sequences,
                    save_checkpoint=save_checkpoint,
                    verbose=True,
                )
                training_time_s = time.perf_counter() - training_started
                if int(result.step) != target_step:
                    raise RuntimeError(
                        f"ART step mismatch: expected {target_step}, got {result.step}."
                    )
                state["current_step"] = target_step
                cast(list[dict[str, Any]], state["updates"]).append(
                    {
                        "step": target_step,
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "reward_mean": audit.mean,
                        "reward_std": audit.std,
                        "rewards": list(audit.rewards),
                        "rollout_time_s": rollout_time_s,
                        "training_time_s": training_time_s,
                        "trainer_metrics": dict(result.metrics),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "update_complete",
                            "step": target_step,
                            "task_id": task.task_id,
                            "reward_mean": audit.mean,
                            "reward_std": audit.std,
                            "rollout_time_s": rollout_time_s,
                            "training_time_s": training_time_s,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                write_json(output_dir / "training_state.json", state)

                if target_step % config.validate_every != 0:
                    continue
                checkpoint_path = Path(
                    get_step_checkpoint_dir(str(model_dir), target_step)
                )
                if not checkpoint_path.exists():
                    raise RuntimeError(f"Milestone checkpoint missing: {checkpoint_path}")
                cast(list[dict[str, Any]], state["checkpoints"]).append(
                    {"step": target_step, "path": str(checkpoint_path)}
                )
                validation_started = time.perf_counter()
                validation_metrics = _validation_metrics(
                    model=model,
                    tasks=validation_tasks,
                    config=config,
                    payload=payload,
                    output_dir=output_dir / "validation" / f"step-{target_step:04d}",
                )
                overall = cast(dict[str, Any], validation_metrics["overall"])
                validation_entry = {
                    "step": target_step,
                    "elapsed_s": time.perf_counter() - validation_started,
                    "overall": overall,
                    "by_task_type": validation_metrics["by_task_type"],
                }
                cast(list[dict[str, Any]], state["validations"]).append(
                    validation_entry
                )
                if best_metrics is None or checkpoint_rank(overall) > checkpoint_rank(
                    best_metrics
                ):
                    best_metrics = overall
                    state["best_step"] = target_step
                validations = cast(list[dict[str, Any]], state["validations"])
                previous = (
                    cast(dict[str, Any], validations[-2]["overall"])
                    if len(validations) > 1
                    else baseline
                )
                reasons = stop_reasons(previous, overall)
                if reasons:
                    state["status"] = "stopped_by_validation_guard"
                    state["stop_reasons"] = reasons
                write_json(output_dir / "training_state.json", state)
                _write_report(
                    output_dir / "CP7_TRAINING_REPORT.md",
                    manifest=manifest,
                    state=state,
                    baseline=baseline,
                )
                print(
                    json.dumps(
                        {
                            "event": "validation_complete",
                            "step": target_step,
                            "task_success": overall["task_success"],
                            "grounded_answer_rate": overall[
                                "grounded_answer_rate"
                            ],
                            "reward_mean": overall["reward_mean"],
                            "best_step": state["best_step"],
                            "stop_reasons": reasons,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if reasons:
                    break

    if state["status"] == "running":
        state["status"] = "complete"
    state["peak_gpu_memory_mib_nvidia_smi"] = gpu.peak_mib
    state["finished_at"] = datetime.now(UTC).isoformat()
    if model_dir is not None:
        state["model_dir"] = str(model_dir)
    write_json(output_dir / "training_state.json", state)
    _write_report(
        output_dir / "CP7_TRAINING_REPORT.md",
        manifest=manifest,
        state=state,
        baseline=baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cp7_grpo_formal.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(train(args.config, args.output))


if __name__ == "__main__":
    main()
