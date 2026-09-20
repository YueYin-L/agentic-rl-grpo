"""Re-run the deterministic verifier over persisted baseline trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from research_router_opt.analysis_models import (
    AgentState,
    AnalysisTrajectory,
    ToolCall,
    ToolResult,
)
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.baseline import (
    EvaluatedTrajectory,
    _agentic_behavior,
    _aggregate,
    _classify_failures,
    _compute_statistics,
    _potential_shortcut,
    _write_report,
)
from research_router_opt.verifier import verify_trajectory


def _git_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _trajectory(payload: dict[str, Any]) -> AnalysisTrajectory:
    raw_state = cast(dict[str, Any], payload["state"])
    calls = [ToolCall(**call) for call in raw_state["tool_calls"]]
    results = [ToolResult(**result) for result in raw_state["tool_results"]]
    state = AgentState(
        task_id=str(raw_state["task_id"]),
        messages=[],
        state=str(raw_state["state"]),
        tool_calls=calls,
        tool_results=results,
        errors=[str(error) for error in raw_state["errors"]],
        step_count=int(raw_state["step_count"]),
        final_answer=raw_state["final_answer"],
        termination_reason=raw_state["termination_reason"],
    )
    return AnalysisTrajectory(
        task_id=str(payload["task_id"]),
        backend_name=str(payload["backend_name"]),
        state=state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cp4-status", default="CP4 PASS after manual trajectory audit")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    source_trajectory_path = args.source / "trajectories.jsonl"
    raw_lines = source_trajectory_path.read_text(encoding="utf-8").splitlines()
    raw_records = [json.loads(line) for line in raw_lines]
    tasks = {task.task_id: task for task in load_analysis_tasks(args.dataset)}
    records: list[EvaluatedTrajectory] = []
    grouped: dict[str, list[EvaluatedTrajectory]] = defaultdict(list)

    with (args.output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for raw in raw_records:
            task_id = str(raw["task"]["task_id"])
            task = tasks[task_id]
            trajectory = _trajectory(cast(dict[str, Any], raw["trajectory"]))
            verification = verify_trajectory(task, trajectory)
            record = EvaluatedTrajectory(
                task=task,
                trajectory=trajectory,
                verification=verification,
                elapsed_s=float(raw["elapsed_s"]),
            )
            records.append(record)
            grouped[task.task_type].append(record)
            raw["verification"] = verification.to_dict()
            raw["failure_modes"] = sorted(_classify_failures(record))
            raw["potential_shortcut"] = _potential_shortcut(record)
            handle.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")

    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    project_root = Path.cwd()
    manifest = dict(source_manifest)
    manifest["rollout_source"] = {
        "run_id": source_manifest["run_id"],
        "git": source_manifest["git"],
        "trajectories_sha256": hashlib.sha256(source_trajectory_path.read_bytes()).hexdigest(),
    }
    manifest["run_id"] = args.output.name
    manifest["verification"] = {
        "git_commit": _git_commit(project_root),
        "rescored_at": datetime.now(UTC).isoformat(),
    }

    summary: dict[str, Any] = {
        "overall": _aggregate(records),
        "by_task_type": {
            task_type: _aggregate(task_records)
            for task_type, task_records in sorted(grouped.items())
        },
        "agentic_behavior": _agentic_behavior(records),
        "compute": _compute_statistics(records),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.output / "BASELINE_REPORT.md",
        manifest=manifest,
        metrics=summary,
        behavior=cast(dict[str, Any], summary["agentic_behavior"]),
        records=records,
        cp4_status=args.cp4_status,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
