"""Canonical validation-baseline runner for an OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast
from urllib.request import Request, urlopen

from research_router_opt.analysis_models import (
    AnalysisTask,
    AnalysisTrajectory,
    VerificationResult,
)
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.backend import ModelBackend, VLLMBackend
from research_router_opt.environment import StatefulToolEnvironment
from research_router_opt.runtime import MultiTurnAgentRuntime, RuntimeConfig
from research_router_opt.verifier import verify_trajectory


@dataclass(frozen=True)
class EvaluatedTrajectory:
    task: AnalysisTask
    trajectory: AnalysisTrajectory
    verification: VerificationResult
    elapsed_s: float


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _has_timeout(record: EvaluatedTrajectory) -> bool:
    state = record.trajectory.state
    return state.termination_reason == "model_timeout" or any(
        result.status == "timeout" for result in state.tool_results
    )


def _has_tool_error(record: EvaluatedTrajectory) -> bool:
    return any(result.status != "ok" for result in record.trajectory.state.tool_results)


def _tool_names(record: EvaluatedTrajectory, *, only_ok: bool = False) -> list[str]:
    if only_ok:
        return [
            result.name
            for result in record.trajectory.state.tool_results
            if result.status == "ok"
        ]
    return [call.name for call in record.trajectory.state.tool_calls]


def _sql_queries(record: EvaluatedTrajectory, *, only_ok: bool = False) -> list[str]:
    successful_ids = {
        result.call_id
        for result in record.trajectory.state.tool_results
        if result.name == "sql" and result.status == "ok"
    }
    return [
        str(call.arguments.get("query", ""))
        for call in record.trajectory.state.tool_calls
        if call.name == "sql" and (not only_ok or call.call_id in successful_ids)
    ]


def _potential_shortcut(record: EvaluatedTrajectory) -> bool:
    if not record.verification.success:
        return False
    task_type = record.task.task_type
    names = _tool_names(record, only_ok=True)
    queries = _sql_queries(record, only_ok=True)
    if task_type == "schema_discovery":
        return "schema" not in names
    if task_type == "multi_table_reasoning":
        return not any(query.casefold().count("join") >= 2 for query in queries)
    if task_type in {"sql_calculator", "tool_selection"}:
        return "sql" not in names or "calculator" not in names
    if task_type == "multi_query_aggregation":
        return names.count("sql") < 2 or "calculator" not in names
    if task_type == "distractor_schema":
        return any("marketing_campaigns" in query.casefold() for query in queries)
    return False


def _classify_failures(record: EvaluatedTrajectory) -> set[str]:
    state = record.trajectory.state
    verification = record.verification
    errors = "\n".join(state.errors).casefold()
    failures: set[str] = set()
    if "table" in errors and any(
        token in errors for token in ("not exist", "not found", "missing")
    ):
        failures.add("wrong_table")
    if "column" in errors or "referenced column" in errors:
        failures.add("wrong_column")
    if any(result.name == "sql" and result.status == "error" for result in state.tool_results):
        failures.add("sql_error")
    if "join" in errors or (
        record.task.task_type == "multi_table_reasoning"
        and verification.sql_exec_success
        and not verification.result_correct
    ):
        failures.add("join_error")
    if any(
        result.name == "calculator" and result.status == "error"
        for result in state.tool_results
    ):
        failures.add("arithmetic_error")
    if state.final_answer is not None and not state.tool_calls:
        failures.add("premature_final")
    if "recovery_failure" in verification.failure_modes:
        failures.add("recovery_failure")
    if state.termination_reason == "max_steps":
        failures.add("tool_loop")
    if state.final_answer is not None and not verification.answer_correct:
        failures.add("hallucination")
    if not verification.tool_valid:
        failures.add("invalid_tool_call")
    if not verification.grounded:
        failures.add("grounding_failure")
    if (
        verification.redundant_calls > 0
        or len(state.tool_calls) > len(record.task.expected_tools) + 2
    ):
        failures.add("excessive_tool_use")
    if _potential_shortcut(record):
        failures.add("potential_shortcut")
    return failures


def _aggregate(records: Sequence[EvaluatedTrajectory]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one evaluated trajectory is required.")
    total = len(records)
    recovery_opportunities = [record for record in records if _has_tool_error(record)]
    tool_calls = [len(record.trajectory.state.tool_calls) for record in records]
    failure_modes = Counter(
        mode for record in records for mode in _classify_failures(record)
    )
    return {
        "task_count": total,
        "task_success": sum(record.verification.success for record in records) / total,
        "final_answer_accuracy": (
            sum(record.verification.answer_correct for record in records) / total
        ),
        "sql_execution_success": (
            sum(record.verification.sql_exec_success for record in records) / total
        ),
        "result_correctness": (
            sum(record.verification.result_correct for record in records) / total
        ),
        "tool_validity": sum(record.verification.tool_valid for record in records) / total,
        "invalid_call_rate": (
            sum(not record.verification.tool_valid for record in records) / total
        ),
        "grounded_answer_rate": (
            sum(record.verification.grounded for record in records) / total
        ),
        "recovery_opportunities": len(recovery_opportunities),
        "recovery_rate": (
            sum(record.verification.recovered_from_error for record in recovery_opportunities)
            / len(recovery_opportunities)
            if recovery_opportunities
            else 0.0
        ),
        "average_steps": (
            sum(record.verification.step_count for record in records) / total
        ),
        "average_tool_calls": sum(tool_calls) / total,
        "timeout_rate": sum(_has_timeout(record) for record in records) / total,
        "failure_modes": dict(sorted(failure_modes.items())),
    }


def _agentic_behavior(records: Sequence[EvaluatedTrajectory]) -> dict[str, Any]:
    checks: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        task_type = record.task.task_type
        names = _tool_names(record, only_ok=True)
        queries = _sql_queries(record, only_ok=True)
        if task_type == "schema_discovery":
            calls = _tool_names(record)
            checks["schema_before_sql"].append(
                "schema" in calls
                and "sql" in calls
                and calls.index("schema") < calls.index("sql")
            )
        if task_type == "error_recovery":
            checks["error_recovery_observed"].append(
                record.verification.recovered_from_error
            )
        if task_type == "multi_table_reasoning":
            checks["multi_table_join"].append(
                any(query.casefold().count("join") >= 2 for query in queries)
            )
        if task_type in {"sql_calculator", "tool_selection"}:
            checks["sql_calculator_composition"].append(
                "sql" in names and "calculator" in names
            )
        if task_type == "multi_query_aggregation":
            checks["multiple_observations"].append(
                names.count("sql") >= 2 and "calculator" in names
            )
        checks["premature_final"].append(
            record.trajectory.state.final_answer is not None
            and not record.trajectory.state.tool_calls
        )
        checks["looping"].append(
            record.trajectory.state.termination_reason == "max_steps"
        )
        checks["potential_shortcut"].append(_potential_shortcut(record))
    return {
        name: {"count": len(values), "rate": sum(values) / len(values)}
        for name, values in sorted(checks.items())
        if values
    }


def _compute_statistics(records: Sequence[EvaluatedTrajectory]) -> dict[str, Any]:
    episode_seconds = [record.elapsed_s for record in records]
    tool_latencies = [
        result.latency_ms
        for record in records
        for result in record.trajectory.state.tool_results
    ]
    total_seconds = sum(episode_seconds)
    return {
        "total_episode_seconds": total_seconds,
        "tasks_per_minute": len(records) / total_seconds * 60 if total_seconds else 0.0,
        "episode_p50_s": median(episode_seconds) if episode_seconds else 0.0,
        "episode_p95_s": _percentile(episode_seconds, 0.95),
        "tool_latency_p50_ms": median(tool_latencies) if tool_latencies else 0.0,
        "tool_latency_p95_ms": _percentile(tool_latencies, 0.95),
    }


def _representatives(records: Sequence[EvaluatedTrajectory]) -> list[EvaluatedTrajectory]:
    selected: list[EvaluatedTrajectory] = []
    groups: Iterable[Iterable[EvaluatedTrajectory]] = (
        (record for record in records if record.verification.success),
        (record for record in records if not record.verification.success),
        (record for record in records if _potential_shortcut(record)),
    )
    for group in groups:
        candidate = next(iter(group), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    return selected


def _candidate_signals(failure_modes: dict[str, int]) -> list[str]:
    candidates: list[str] = []
    if any(failure_modes.get(name, 0) for name in ("wrong_table", "wrong_column")):
        candidates.append("schema_valid and result_correct for table/column selection failures")
    if any(failure_modes.get(name, 0) for name in ("sql_error", "join_error")):
        candidates.append("sql_exec_success and result_correct for executable relational reasoning")
    if failure_modes.get("arithmetic_error", 0):
        candidates.append("answer_correct after a valid calculator observation")
    if failure_modes.get("recovery_failure", 0):
        candidates.append("recovered_from_error for post-error correction")
    if any(
        failure_modes.get(name, 0)
        for name in ("premature_final", "hallucination", "grounding_failure")
    ):
        candidates.append("grounded with answer_correct to penalize unsupported final answers")
    if any(
        failure_modes.get(name, 0)
        for name in ("tool_loop", "excessive_tool_use")
    ):
        candidates.append("redundant_calls and bounded step_count as efficiency penalties")
    if failure_modes.get("invalid_tool_call", 0):
        candidates.append("tool_valid for parser/schema-conformant actions")
    if failure_modes.get("potential_shortcut", 0):
        candidates.append("task-type behavior checks to prevent outcome-only shortcut optimization")
    return candidates


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    behavior: dict[str, Any],
    records: Sequence[EvaluatedTrajectory],
    cp4_status: str,
) -> None:
    overall = cast(dict[str, Any], metrics["overall"])
    lines = [
        "# Canonical Agent Baseline Report",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in overall.items():
        if name != "failure_modes":
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {name} | {rendered} |")
    lines.extend(("", "## Metrics by Task Type", ""))
    for task_type, task_metrics in cast(dict[str, dict[str, Any]], metrics["by_task_type"]).items():
        lines.extend(
            (
                f"### {task_type}",
                "",
                f"- task_count: {task_metrics['task_count']}",
                f"- task_success: {task_metrics['task_success']:.4f}",
                f"- final_answer_accuracy: {task_metrics['final_answer_accuracy']:.4f}",
                f"- grounded_answer_rate: {task_metrics['grounded_answer_rate']:.4f}",
                f"- average_tool_calls: {task_metrics['average_tool_calls']:.4f}",
                "",
            )
        )
    failure_modes = cast(dict[str, int], overall["failure_modes"])
    lines.extend(("## Failure Modes", ""))
    if failure_modes:
        lines.extend(f"- {name}: {count}" for name, count in failure_modes.items())
    else:
        lines.append("- No programmatic failure mode was observed.")
    lines.extend(("", "## Agentic Behavior Analysis", ""))
    lines.extend(
        f"- {name}: {entry['count']} tasks, rate={entry['rate']:.4f}"
        for name, entry in behavior.items()
    )
    lines.extend(("", "## Representative Trajectories", ""))
    for record in _representatives(records):
        calls = [
            f"{result.name}:{result.status}"
            for result in record.trajectory.state.tool_results
        ]
        lines.extend(
            (
                f"### {record.task.task_id}",
                "",
                f"- question: {record.task.question}",
                f"- tools: {', '.join(calls) or 'none'}",
                f"- final_answer: {record.trajectory.state.final_answer}",
                f"- success: {record.verification.success}",
                f"- failure_modes: {', '.join(sorted(_classify_failures(record))) or 'none'}",
                "",
            )
        )
    lines.extend(("## Candidate Reward Signals", ""))
    candidates = _candidate_signals(failure_modes)
    lines.extend(f"- {candidate}" for candidate in candidates)
    if not candidates:
        lines.append(
            "- No reward signal should be frozen before a non-degenerate failure set exists."
        )
    lines.extend(
        (
            "",
            "No weights are assigned at CP4; CP5 must audit these candidates offline.",
            "",
            "## Compute Statistics",
            "",
            "```json",
            json.dumps(metrics["compute"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## CP4 Status",
            "",
            cp4_status,
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline(
    *,
    tasks: Sequence[AnalysisTask],
    backend: ModelBackend,
    output_dir: Path,
    runtime_config: RuntimeConfig | None = None,
    manifest: dict[str, Any] | None = None,
    cp4_status: str = "PENDING_MANUAL_REVIEW",
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("At least one validation task is required.")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[EvaluatedTrajectory] = []
    grouped: dict[str, list[EvaluatedTrajectory]] = defaultdict(list)
    trajectory_path = output_dir / "trajectories.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle, StatefulToolEnvironment() as env:
        runtime = MultiTurnAgentRuntime(backend, env, runtime_config)
        for task in tasks:
            started = perf_counter()
            trajectory = runtime.run(task)
            elapsed_s = perf_counter() - started
            verification = verify_trajectory(task, trajectory)
            record = EvaluatedTrajectory(task, trajectory, verification, elapsed_s)
            records.append(record)
            grouped[task.task_type].append(record)
            handle.write(
                json.dumps(
                    {
                        "task": task.to_dict(),
                        "trajectory": trajectory.to_dict(),
                        "verification": verification.to_dict(),
                        "failure_modes": sorted(_classify_failures(record)),
                        "potential_shortcut": _potential_shortcut(record),
                        "elapsed_s": elapsed_s,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            handle.flush()

    summary: dict[str, Any] = {
        "overall": _aggregate(records),
        "by_task_type": {
            task_type: _aggregate(task_records)
            for task_type, task_records in sorted(grouped.items())
        },
        "agentic_behavior": _agentic_behavior(records),
        "compute": _compute_statistics(records),
    }
    effective_manifest = dict(manifest or {})
    effective_manifest["task_ids_sha256"] = hashlib.sha256(
        "\n".join(task.task_id for task in tasks).encode("utf-8")
    ).hexdigest()
    effective_manifest["task_count"] = len(tasks)
    (output_dir / "manifest.json").write_text(
        json.dumps(effective_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "BASELINE_REPORT.md",
        manifest=effective_manifest,
        metrics=summary,
        behavior=cast(dict[str, Any], summary["agentic_behavior"]),
        records=records,
        cp4_status=cp4_status,
    )
    return summary


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "working_tree_clean": not status,
        "status": status,
    }


def _served_models(base_url: str, api_key: str, timeout_s: float) -> list[str]:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return [str(entry["id"]) for entry in payload.get("data", [])]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cp4_qwen35_9b.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--phase", choices=("smoke", "canonical"), default="smoke")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    config = _load_config(args.config)
    project_root = Path.cwd()
    git = _git_metadata(project_root)
    if not git["working_tree_clean"] and not args.allow_dirty:
        parser.error("Git working tree must be clean; commit configuration before the baseline")
    model_config = cast(dict[str, Any], config["model"])
    runtime_values = cast(dict[str, Any], config["runtime"])
    dataset_config = cast(dict[str, Any], config["dataset"])
    data_path = project_root / str(dataset_config["validation_path"])
    actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if actual_hash != dataset_config["validation_sha256"]:
        parser.error("Validation dataset hash does not match the frozen CP4 configuration")

    model = str(model_config["id"])
    base_url = os.environ.get("VLLM_BASE_URL", "").strip()
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    if not base_url:
        parser.error("VLLM_BASE_URL must point to the verified Linux vLLM /v1 endpoint")
    served_models = _served_models(base_url, api_key, float(runtime_values["model_timeout_s"]))
    if model not in served_models:
        parser.error(
            f"Configured model {model!r} is not served; "
            f"endpoint returned {served_models!r}"
        )

    tasks = load_analysis_tasks(data_path)[: args.limit]
    started_at = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "run_id": args.output.name,
        "phase": args.phase,
        "started_at": started_at.isoformat(),
        "git": git,
        "model": model_config,
        "runtime": runtime_values,
        "vllm": config["vllm"],
        "dataset": {
            "path": str(data_path),
            "sha256": actual_hash,
            "split": "validation",
        },
        "endpoint": {"base_url": base_url, "served_models": served_models},
        "compute": {
            "gpu": os.environ.get("BASELINE_GPU_NAME", "UNRECORDED"),
            "gpu_vram": os.environ.get("BASELINE_GPU_VRAM", "UNRECORDED"),
            "driver": os.environ.get("BASELINE_GPU_DRIVER", "UNRECORDED"),
        },
    }
    backend = VLLMBackend(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=int(runtime_values["max_output_tokens"]),
    )
    summary = run_baseline(
        tasks=tasks,
        backend=backend,
        output_dir=args.output,
        runtime_config=RuntimeConfig(
            max_steps=int(runtime_values["max_steps"]),
            model_timeout_s=float(runtime_values["model_timeout_s"]),
            tool_timeout_s=float(runtime_values["tool_timeout_s"]),
        ),
        manifest=manifest,
        cp4_status=f"{args.phase.upper()} PENDING MANUAL AGENTIC-BEHAVIOR REVIEW",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
