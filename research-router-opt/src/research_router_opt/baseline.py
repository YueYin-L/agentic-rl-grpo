"""Canonical validation-baseline runner for an OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from research_router_opt.analysis_models import AnalysisTask, VerificationResult
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.backend import ModelBackend, VLLMBackend
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig
from research_router_opt.verifier import verify_trajectory


def _aggregate(results: Sequence[tuple[AnalysisTask, VerificationResult]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one verification result is required.")
    total = len(results)
    metric_names = (
        "success",
        "answer_correct",
        "sql_exec_success",
        "result_correct",
        "tool_valid",
        "schema_valid",
        "grounded",
        "recovered_from_error",
    )
    metrics: dict[str, Any] = {
        name: sum(bool(getattr(result, name)) for _, result in results) / total
        for name in metric_names
    }
    metrics["task_count"] = total
    metrics["average_steps"] = sum(result.step_count for _, result in results) / total
    metrics["average_redundant_calls"] = (
        sum(result.redundant_calls for _, result in results) / total
    )
    failure_modes = Counter(
        mode for _, result in results for mode in result.failure_modes
    )
    metrics["failure_modes"] = dict(sorted(failure_modes.items()))
    return metrics


def run_baseline(
    *,
    tasks: Sequence[AnalysisTask],
    backend: ModelBackend,
    output_dir: Path,
    runtime_config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("At least one validation task is required.")
    output_dir.mkdir(parents=True, exist_ok=False)
    verified: list[tuple[AnalysisTask, VerificationResult]] = []
    grouped: dict[str, list[tuple[AnalysisTask, VerificationResult]]] = defaultdict(list)
    trajectory_path = output_dir / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle, StatefulToolEnvironment() as env:
        runtime = MultiTurnAgentRuntime(backend, env, runtime_config)
        for task in tasks:
            trajectory = runtime.run(task)
            verification = verify_trajectory(task, trajectory)
            verified.append((task, verification))
            grouped[task.task_type].append((task, verification))
            handle.write(
                json.dumps(
                    {
                        "task": task.to_dict(),
                        "trajectory": trajectory.to_dict(),
                        "verification": verification.to_dict(),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            handle.flush()

    summary = {
        "overall": _aggregate(verified),
        "by_task_type": {
            task_type: _aggregate(task_results)
            for task_type, task_results in sorted(grouped.items())
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/analysis_validation.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    model = os.environ.get("VLLM_MODEL", "").strip()
    base_url = os.environ.get("VLLM_BASE_URL", "").strip()
    if not model or not base_url:
        parser.error("VLLM_MODEL and VLLM_BASE_URL must be set to verified values")
    backend = VLLMBackend(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    tasks = load_analysis_tasks(args.data)[: args.limit]
    summary = run_baseline(
        tasks=tasks,
        backend=backend,
        output_dir=args.output,
        runtime_config=RuntimeConfig(max_steps=args.max_steps),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
