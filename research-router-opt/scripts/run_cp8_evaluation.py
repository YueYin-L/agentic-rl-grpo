"""Freeze and execute the one-shot CP8 Base-vs-GRPO frozen-test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from run_cp6_smoke import GpuMonitor

from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.backend import VLLMBackend
from research_router_opt.baseline import (
    EvaluatedTrajectory,
    _agentic_behavior,
    _aggregate,
    _classify_failures,
    _compute_statistics,
    _potential_shortcut,
)
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.final_evaluation import (
    exact_mcnemar_p_value,
    metric_deltas,
    paired_transitions,
)
from research_router_opt.reward import load_reward_config, score_agent_trajectory
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig
from research_router_opt.verifier import verify_trajectory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _git(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--porcelain")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "status": status,
        "tracked_status": tracked_status,
    }


def _served_models(base_url: str, timeout_s: float) -> list[str]:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return [str(entry["id"]) for entry in payload["data"]]


def _command_output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def freeze_gate(config_path: Path, output: Path, endpoint: str) -> None:
    root = Path.cwd()
    config = _read_config(config_path)
    model = cast(dict[str, Any], config["model"])
    dataset = cast(dict[str, Any], config["dataset"])
    runtime = cast(dict[str, Any], config["runtime"])
    inference = cast(dict[str, Any], config["inference"])
    verifier = cast(dict[str, Any], config["verifier"])
    adapter_path = Path(str(model["adapter_path"]))
    test_path = root / str(dataset["test_path"])
    adapter_file = adapter_path / "adapter_model.safetensors"
    if _sha256(adapter_file) != model["adapter_sha256"]:
        raise RuntimeError("Step-80 adapter hash mismatch; CP8 is stopped.")
    if _sha256(test_path) != dataset["test_sha256"]:
        raise RuntimeError("Frozen-test hash mismatch; CP8 is stopped.")
    tasks = load_analysis_tasks(test_path)
    if len(tasks) != int(dataset["task_count"]) or any(task.split != "test" for task in tasks):
        raise RuntimeError("Frozen-test count/split mismatch; CP8 is stopped.")
    if _sha256(root / str(verifier["reward_config"])) != verifier["reward_sha256"]:
        raise RuntimeError("Frozen reward hash mismatch; CP8 is stopped.")
    served = _served_models(endpoint, float(runtime["model_timeout_s"]))
    expected_models = {str(model["base_id"]), str(model["grpo_served_name"])}
    if not expected_models.issubset(set(served)):
        raise RuntimeError(f"Endpoint models {served!r} do not expose {sorted(expected_models)!r}.")
    git = _git(root)
    if git["tracked_dirty"]:
        raise RuntimeError(f"Tracked working tree is dirty before Gate C:\n{git['tracked_status']}")
    gpu = _command_output(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    )
    manifest = {
        "run_id": config["run"]["id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "git": git,
        "base_model_id": model["base_id"],
        "base_revision": model["revision"],
        "precision": model["precision"],
        "quantization": model["quantization"],
        "grpo_checkpoint_path": str(adapter_path),
        "grpo_adapter_sha256": model["adapter_sha256"],
        "frozen_test_path": str(test_path),
        "frozen_test_sha256": dataset["test_sha256"],
        "test_task_count": len(tasks),
        "verifier": {
            "implementation": verifier["implementation"],
            "source_sha256": _sha256(root / "src/research_router_opt/verifier.py"),
            "reward_config": verifier["reward_config"],
            "reward_sha256": verifier["reward_sha256"],
            "git_commit": git["commit"],
        },
        "runtime_config": runtime,
        "inference_config": inference,
        "prompt_identity": {
            "runtime_source_sha256": _sha256(root / "src/research_router_opt/runtime.py"),
            "chat_template": model["chat_template"],
        },
        "tool_parser": model["tool_call_parser"],
        "endpoint": endpoint,
        "served_models": served,
        "vllm_version": os.environ.get("CP8_VLLM_VERSION", "0.25.1"),
        "cuda_gpu": gpu,
        "test_used_before_cp8": False,
        "retraining_after_test": "forbidden",
        "verifier_changes_after_test": "forbidden",
        "checkpoint_selection_after_test": "forbidden",
        "fairness": (
            "same vLLM process, parser, prompt, tools, runtime, verifier, seed, and "
            "generation parameters; only LoRA policy differs"
        ),
    }
    manifest_path = output / "gate_c_manifest.json"
    _write_json(manifest_path, manifest, exclusive=True)
    (output / "gate_c_manifest.sha256").write_text(_sha256(manifest_path) + "\n", encoding="utf-8")


def _verify_gate(config: dict[str, Any], output: Path) -> dict[str, Any]:
    manifest_path = output / "gate_c_manifest.json"
    expected = (output / "gate_c_manifest.sha256").read_text(encoding="utf-8").strip()
    if _sha256(manifest_path) != expected:
        raise RuntimeError("Gate C manifest changed after protocol freeze.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["git"]["commit"] != _git(Path.cwd())["commit"]:
        raise RuntimeError("Git commit changed after Gate C freeze.")
    model = config["model"]
    dataset = config["dataset"]
    if (
        _sha256(Path(str(model["adapter_path"])) / "adapter_model.safetensors")
        != model["adapter_sha256"]
    ):
        raise RuntimeError("Adapter changed after Gate C freeze.")
    if _sha256(Path(str(dataset["test_path"]))) != dataset["test_sha256"]:
        raise RuntimeError("Frozen test changed after Gate C freeze.")
    return cast(dict[str, Any], manifest)


def evaluate_model(config_path: Path, output: Path, endpoint: str, policy: str) -> None:
    config = _read_config(config_path)
    gate = _verify_gate(config, output)
    model_config = cast(dict[str, Any], config["model"])
    runtime_values = cast(dict[str, Any], config["runtime"])
    inference = cast(dict[str, Any], config["inference"])
    dataset = cast(dict[str, Any], config["dataset"])
    model_name = (
        str(model_config["base_id"]) if policy == "base" else str(model_config["grpo_served_name"])
    )
    if model_name not in _served_models(endpoint, float(runtime_values["model_timeout_s"])):
        raise RuntimeError(f"Frozen policy model {model_name!r} is not served.")
    policy_dir = output / ("base" if policy == "base" else "grpo-step80")
    policy_dir.mkdir(parents=True, exist_ok=False)
    tasks = load_analysis_tasks(Path(str(dataset["test_path"])))
    reward_config = load_reward_config(Path(str(config["verifier"]["reward_config"])))
    backend = VLLMBackend(
        base_url=endpoint,
        model=model_name,
        max_tokens=int(inference["max_output_tokens"]),
        seed=int(inference["seed"]),
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
    )
    runtime_config = RuntimeConfig(
        max_steps=int(runtime_values["max_steps"]),
        model_timeout_s=float(runtime_values["model_timeout_s"]),
        tool_timeout_s=float(runtime_values["tool_timeout_s"]),
    )
    records: list[EvaluatedTrajectory] = []
    raw_records: list[dict[str, Any]] = []
    grouped: dict[str, list[EvaluatedTrajectory]] = defaultdict(list)
    rewards: list[float] = []
    started = time.perf_counter()
    with (
        GpuMonitor() as gpu,
        (policy_dir / "trajectories.jsonl").open("x", encoding="utf-8") as handle,
        StatefulToolEnvironment() as environment,
    ):
        runtime = MultiTurnAgentRuntime(backend, environment, runtime_config)
        for task in tasks:
            task_started = time.perf_counter()
            trajectory = runtime.run(task)
            elapsed_s = time.perf_counter() - task_started
            if trajectory.state.termination_reason in {"model_error", "model_timeout"}:
                failure = {
                    "task_id": task.task_id,
                    "termination_reason": trajectory.state.termination_reason,
                    "errors": trajectory.state.errors,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                _write_json(policy_dir / "infrastructure_failure.json", failure)
                raise RuntimeError(
                    f"Infrastructure failure on {task.task_id}; model retry not performed."
                )
            verification = verify_trajectory(task, trajectory)
            reward = score_agent_trajectory(task, trajectory, verification, reward_config)
            record = EvaluatedTrajectory(task, trajectory, verification, elapsed_s)
            records.append(record)
            grouped[task.task_type].append(record)
            rewards.append(reward.total_reward)
            raw = {
                "task": task.to_dict(),
                "model_identity": model_name,
                "trajectory": trajectory.to_dict(),
                "verification": verification.to_dict(),
                "reward": reward.to_dict(),
                "failure_modes": sorted(_classify_failures(record)),
                "potential_shortcut": _potential_shortcut(record),
                "elapsed_s": elapsed_s,
            }
            raw_records.append(raw)
            handle.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")
            handle.flush()
    total_s = time.perf_counter() - started
    overall = _aggregate(records)
    overall.update(
        {
            "max_step_termination_rate": sum(
                item.trajectory.state.termination_reason == "max_steps" for item in records
            )
            / len(records),
            "reward_mean": statistics.mean(rewards),
            "reward_std": statistics.pstdev(rewards),
        }
    )
    metrics = {
        "overall": overall,
        "by_task_type": {name: _aggregate(values) for name, values in sorted(grouped.items())},
        "agentic_behavior": _agentic_behavior(records),
        "compute": {
            **_compute_statistics(records),
            "wall_time_s": total_s,
            "peak_gpu_memory_mib_nvidia_smi": gpu.peak_mib,
            "trajectory_count": len(records),
            "failed_infrastructure_calls": 0,
            "resume_retry_events": 0,
        },
    }
    _write_json(policy_dir / "metrics.json", metrics)
    _write_json(
        policy_dir / "manifest.json",
        {
            "gate_c_manifest_sha256": _sha256(output / "gate_c_manifest.json"),
            "policy": policy,
            "model_identity": model_name,
            "task_count": len(records),
            "task_ids_sha256": hashlib.sha256(
                "\n".join(item.task.task_id for item in records).encode()
            ).hexdigest(),
            "started_after_gate_timestamp": gate["timestamp"],
            "completed_at": datetime.now(UTC).isoformat(),
            "infrastructure_retries": 0,
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare(output: Path) -> None:
    base_records = _read_jsonl(output / "base/trajectories.jsonl")
    grpo_records = _read_jsonl(output / "grpo-step80/trajectories.jsonl")
    if len(base_records) != 150 or len(grpo_records) != 150:
        raise RuntimeError("CP8 requires exactly 150 canonical trajectories per policy.")
    base_metrics = json.loads((output / "base/metrics.json").read_text(encoding="utf-8"))
    grpo_metrics = json.loads((output / "grpo-step80/metrics.json").read_text(encoding="utf-8"))
    transitions = paired_transitions(base_records, grpo_records)
    counts = transitions["overall"]
    p_value = exact_mcnemar_p_value(
        counts["base_success_grpo_fail"], counts["base_fail_grpo_success"]
    )
    shortcut_ids = [
        record["task"]["task_id"] for record in grpo_records if record["potential_shortcut"]
    ]
    grpo_successes = sum(record["verification"]["success"] for record in grpo_records)
    strict_successes = sum(
        record["verification"]["success"] and not record["potential_shortcut"]
        for record in grpo_records
    )
    by_type: dict[str, Any] = {}
    for task_type in sorted(base_metrics["by_task_type"]):
        base_entry = base_metrics["by_task_type"][task_type]
        grpo_entry = grpo_metrics["by_task_type"][task_type]
        by_type[task_type] = {
            "n": base_entry["task_count"],
            "base_task_success": base_entry["task_success"],
            "grpo_task_success": grpo_entry["task_success"],
            "delta_pp": 100 * (grpo_entry["task_success"] - base_entry["task_success"]),
        }
    comparison = {
        "overall_deltas": metric_deltas(base_metrics["overall"], grpo_metrics["overall"]),
        "by_task_type": by_type,
        "paired_transitions": transitions,
        "statistics": {
            "test": "two-sided exact McNemar",
            "discordant_pairs": counts["base_success_grpo_fail"] + counts["base_fail_grpo_success"],
            "p_value": p_value,
        },
        "shortcut_audit": {
            "potential_shortcut_task_ids": shortcut_ids,
            "potential_shortcut_count": len(shortcut_ids),
            "grpo_success_count": grpo_successes,
            "strict_success_count": strict_successes,
            "strict_task_success": strict_successes / len(grpo_records),
        },
    }
    comparison_dir = output / "comparison"
    comparison_dir.mkdir(exist_ok=False)
    _write_json(comparison_dir / "comparison.json", comparison)
    _write_json(comparison_dir / "paired_transitions.json", transitions)
    _write_report(output / "CP8_FINAL_EVALUATION_REPORT.md", comparison, base_metrics, grpo_metrics)


def _write_report(
    path: Path,
    comparison: dict[str, Any],
    base: dict[str, Any],
    grpo: dict[str, Any],
) -> None:
    deltas = comparison["overall_deltas"]
    transitions = comparison["paired_transitions"]["overall"]
    lines = [
        "# CP8 Frozen Test Base-vs-GRPO Final Evaluation",
        "",
        "## Verdict",
        "",
        "The verdict is determined by the frozen held-out metrics below. No retraining, "
        "verifier change, checkpoint reselection, or model retry occurred after test access.",
        "",
        "## Claim Boundary",
        "",
        "This report compares the original Qwen3.5-9B policy with the frozen CP7 step-80 "
        "LoRA on the same 150 held-out tasks under one vLLM process and one evaluation harness.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Base | GRPO step 80 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, entry in deltas.items():
        lines.append(
            f"| {name} | {entry['base']:.4f} | {entry['grpo']:.4f} | "
            f"{entry['absolute_delta']:+.4f} |"
        )
    lines.extend(
        (
            "",
            "## Metrics by Task Type",
            "",
            "| Task type | N | Base | GRPO | Delta pp |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for task_type, entry in comparison["by_task_type"].items():
        lines.append(
            f"| {task_type} | {entry['n']} | {entry['base_task_success']:.2%} | "
            f"{entry['grpo_task_success']:.2%} | {entry['delta_pp']:+.2f} |"
        )
    lines.extend(
        (
            "",
            "## Paired Transition Analysis",
            "",
            f"- Base fail -> GRPO success: {transitions['base_fail_grpo_success']}",
            f"- Base success -> GRPO fail: {transitions['base_success_grpo_fail']}",
            f"- Both success: {transitions['both_success']}",
            f"- Both fail: {transitions['both_fail']}",
            f"- Exact McNemar p-value: {comparison['statistics']['p_value']:.6g}",
            "",
            "## Shortcut / Reward-Hacking Audit",
            "",
            "- Programmatic potential-shortcut flags: "
            f"{comparison['shortcut_audit']['potential_shortcut_count']}",
            "- Strict GRPO Task Success after excluding flagged successes: "
            f"{comparison['shortcut_audit']['strict_task_success']:.2%}",
            "",
            "## Efficiency and Compute",
            "",
            f"- Base wall time: {base['compute']['wall_time_s']:.1f} s",
            f"- GRPO wall time: {grpo['compute']['wall_time_s']:.1f} s",
            f"- Base peak VRAM: {base['compute']['peak_gpu_memory_mib_nvidia_smi']} MiB",
            f"- GRPO peak VRAM: {grpo['compute']['peak_gpu_memory_mib_nvidia_smi']} MiB",
            "- Infrastructure retries: 0",
            "",
            "## Integrity, Limitations, and Reproduction",
            "",
            "See `gate_c_manifest.json` for exact hashes, commit, model revision, parser, "
            "runtime, and generation parameters. The synthetic task suite has 150 held-out "
            "items; per-type conclusions should not be generalized beyond this environment.",
            "",
            "Reproduction phases: `gate`, `evaluate --policy base`, `evaluate --policy grpo`, "
            "then `compare`. The frozen-test phases must not be rerun to select a better outcome.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("gate", "evaluate", "compare"))
    parser.add_argument("--config", type=Path, default=Path("configs/cp8_frozen_test.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("VLLM_BASE_URL", ""))
    parser.add_argument("--policy", choices=("base", "grpo"))
    args = parser.parse_args()
    if args.phase in {"gate", "evaluate"} and not args.endpoint:
        parser.error("--endpoint or VLLM_BASE_URL is required")
    if args.phase == "gate":
        freeze_gate(args.config, args.output, args.endpoint)
    elif args.phase == "evaluate":
        if args.policy is None:
            parser.error("--policy is required for evaluate")
        evaluate_model(args.config, args.output, args.endpoint, args.policy)
    else:
        compare(args.output)


if __name__ == "__main__":
    main()
