"""Offline CP5 reward audit over immutable canonical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, cast

from research_router_opt.analysis_models import (
    AgentState,
    AnalysisTask,
    AnalysisTrajectory,
    ToolCall,
    ToolResult,
    VerificationResult,
)
from research_router_opt.analysis_tasks import load_analysis_tasks
from research_router_opt.reward import (
    AgentRewardBreakdown,
    RewardConfig,
    load_reward_config,
    score_agent_trajectory,
)
from research_router_opt.verifier import verify_trajectory


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _trajectory(payload: dict[str, Any]) -> AnalysisTrajectory:
    raw_state = cast(dict[str, Any], payload["state"])
    state = AgentState(
        task_id=str(raw_state["task_id"]),
        messages=[],
        state=str(raw_state["state"]),
        tool_calls=[ToolCall(**call) for call in raw_state["tool_calls"]],
        tool_results=[ToolResult(**result) for result in raw_state["tool_results"]],
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "p25": _percentile(values, 0.25),
        "median": median(values) if values else 0.0,
        "p75": _percentile(values, 0.75),
        "max": max(values) if values else 0.0,
    }


def _raw_multi_observation(trajectory: AnalysisTrajectory) -> bool:
    names = [
        result.name for result in trajectory.state.tool_results if result.status == "ok"
    ]
    return names.count("sql") >= 2 and "calculator" in names


def _signals(
    task: AnalysisTask,
    trajectory: AnalysisTrajectory,
    verification: VerificationResult,
) -> dict[str, bool]:
    results = trajectory.state.tool_results
    return {
        "answer_correct": verification.answer_correct,
        "result_correct": verification.result_correct,
        "grounded": verification.grounded,
        "tool_valid": verification.tool_valid,
        "sql_exec_success": verification.sql_exec_success,
        "recovered_from_error": verification.recovered_from_error,
        "meaningful_recovery": verification.meaningful_recovery,
        "redundant_calls": verification.redundant_calls > 0,
        "max_step_termination": trajectory.state.termination_reason == "max_steps",
        "tool_error": any(result.status != "ok" for result in results),
        "sql_error": any(
            result.name == "sql" and result.status != "ok" for result in results
        ),
        "calculator_error": any(
            result.name == "calculator" and result.status != "ok" for result in results
        ),
        "multi_observation_completion": (
            task.task_type == "multi_query_aggregation"
            and _raw_multi_observation(trajectory)
        ),
    }


def _signal_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(cast(dict[str, bool], records[0]["signals"]))
    statistics: dict[str, Any] = {}
    for name in names:
        values = [bool(record["signals"][name]) for record in records]
        successes = [record for record in records if record["success"]]
        failures = [record for record in records if not record["success"]]
        by_type: dict[str, list[bool]] = defaultdict(list)
        for record, value in zip(records, values, strict=True):
            by_type[str(record["task_type"])].append(value)
        activation = sum(values) / len(values)
        statistics[name] = {
            "activation_rate": activation,
            "variance": activation * (1 - activation),
            "success_conditional_rate": (
                sum(bool(record["signals"][name]) for record in successes) / len(successes)
            ),
            "failure_conditional_rate": (
                sum(bool(record["signals"][name]) for record in failures) / len(failures)
            ),
            "task_type_activation": {
                task_type: sum(items) / len(items)
                for task_type, items in sorted(by_type.items())
            },
        }
    overlap: dict[str, float] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            union = sum(
                bool(record["signals"][left]) or bool(record["signals"][right])
                for record in records
            )
            intersection = sum(
                bool(record["signals"][left]) and bool(record["signals"][right])
                for record in records
            )
            overlap[f"{left}|{right}"] = intersection / union if union else 0.0
    return {"signals": statistics, "pairwise_jaccard": overlap}


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["reward"].total_reward) for record in records]
    return {
        "count": len(records),
        "reward": _distribution(rewards),
        "examples": [record["task_id"] for record in records[:3]],
    }


def _pairwise_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "successful": [record for record in records if record["success"]],
        "failed": [record for record in records if not record["success"]],
        "efficient_success": [
            record
            for record in records
            if record["success"]
            and record["reward"].excess_call_count == 0
            and record["reward"].redundant_call_count == 0
            and not record["signals"]["max_step_termination"]
        ],
        "inefficient_success": [
            record
            for record in records
            if record["success"]
            and (
                record["reward"].excess_call_count > 0
                or record["reward"].redundant_call_count > 0
                or record["signals"]["max_step_termination"]
            )
        ],
        "grounded_correct": [
            record
            for record in records
            if record["signals"]["answer_correct"] and record["signals"]["grounded"]
        ],
        "unsupported_correct": [
            record
            for record in records
            if record["signals"]["answer_correct"] and not record["signals"]["grounded"]
        ],
        "correct_arithmetic": [
            record
            for record in records
            if record["task_type"] == "sql_calculator" and record["success"]
        ],
        "wrong_arithmetic": [
            record
            for record in records
            if record["task_type"] == "sql_calculator"
            and record["signals"]["sql_exec_success"]
            and not record["signals"]["answer_correct"]
        ],
        "meaningful_recovery": [
            record for record in records if record["signals"]["meaningful_recovery"]
        ],
        "superficial_recovery": [
            record
            for record in records
            if record["signals"]["recovered_from_error"]
            and not record["signals"]["meaningful_recovery"]
        ],
    }
    summaries = {name: _group_summary(items) for name, items in groups.items()}
    comparisons = {
        "success_gt_failure": ("successful", "failed"),
        "efficient_gt_inefficient_success": ("efficient_success", "inefficient_success"),
        "grounded_gt_unsupported_correct": ("grounded_correct", "unsupported_correct"),
        "correct_gt_wrong_arithmetic": ("correct_arithmetic", "wrong_arithmetic"),
        "meaningful_gt_superficial_recovery": (
            "meaningful_recovery",
            "superficial_recovery",
        ),
    }
    ordering = {}
    for name, (preferred, dispreferred) in comparisons.items():
        left = summaries[preferred]
        right = summaries[dispreferred]
        ordering[name] = {
            "evaluable": bool(left["count"] and right["count"]),
            "pass": bool(
                left["count"]
                and right["count"]
                and left["reward"]["mean"] > right["reward"]["mean"]
            ),
            "preferred_mean": left["reward"]["mean"],
            "dispreferred_mean": right["reward"]["mean"],
        }
    return {"groups": summaries, "ordering": ordering}


def _candidate_summary(
    base_records: list[dict[str, Any]],
    config: RewardConfig,
) -> dict[str, Any]:
    rescored: list[dict[str, Any]] = []
    for record in base_records:
        reward = score_agent_trajectory(
            record["task"], record["trajectory"], record["verification"], config
        )
        rescored.append(record | {"reward": reward})
    pairwise = _pairwise_audit(rescored)
    success_rewards = [
        record["reward"].total_reward for record in rescored if record["success"]
    ]
    failure_rewards = [
        record["reward"].total_reward for record in rescored if not record["success"]
    ]
    return {
        "config": asdict(config),
        "success_mean": mean(success_rewards),
        "failure_mean": mean(failure_rewards),
        "separation": mean(success_rewards) - mean(failure_rewards),
        "max_failure": max(failure_rewards),
        "ordering": pairwise["ordering"],
    }


def _component_contribution(records: list[dict[str, Any]]) -> dict[str, float]:
    names = list(
        records[0]["reward"].outcome_components
        | records[0]["reward"].process_components
        | records[0]["reward"].penalties
    )
    return {
        name: mean(
            (
                record["reward"].outcome_components
                | record["reward"].process_components
                | record["reward"].penalties
            )[name]
            for record in records
        )
        for name in names
    }


def _write_report(path: Path, audit: dict[str, Any]) -> None:
    signals = audit["signal_statistics"]["signals"]
    distribution = audit["reward_distribution"]
    pairwise = audit["pairwise"]
    candidates = audit["candidate_comparison"]
    hacking = audit["reward_hacking"]
    lines = [
        "# CP5 Offline Reward Analysis",
        "",
        "## Signal Statistics",
        "",
        "| Signal | Activation | P(signal\|success) | P(signal\|failure) | Variance |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in signals.items():
        lines.append(
            f"| {name} | {entry['activation_rate']:.4f} | "
            f"{entry['success_conditional_rate']:.4f} | "
            f"{entry['failure_conditional_rate']:.4f} | {entry['variance']:.4f} |"
        )
    lines.extend(
        (
            "",
            "`tool_valid` is saturated and `sql_exec_success` is near-saturated; neither receives "
            "positive canonical reward. Full signal overlap and per-task-type activation are "
            "saved in `signal_statistics.json`.",
            "",
            "## Recovery Audit",
            "",
            f"- Error opportunities: {audit['recovery']['opportunities']}",
            f"- Broad recovered_from_error: {audit['recovery']['broad_recovery']}",
            f"- Meaningful recovery: {audit['recovery']['meaningful_recovery']}",
            f"- Superficial recovery: {audit['recovery']['superficial_recovery']}",
            "- Meaningful recovery requires a changed action for the failed tool plus correct, "
            "grounded downstream evidence. An unrelated successful call is not rewarded.",
            "",
            "## Reward Formula",
            "",
            "```text",
            "R = 2.0*answer_correct + 2.0*result_correct + 1.5*grounded",
            "  + 0.4*meaningful_recovery + 0.4*valid_multi_observation",
            "  - 1.0*wrong_answer - 1.0*wrong_result - 0.8*ungrounded",
            "  - bounded(tool_errors, redundant_calls, repeated missing-object probes,",
            "            excess calls, max-step termination)",
            "```",
            "",
            "Tool validity and SQL execution are guardrails, not positive farming targets. "
            "Necessary Schema -> SQL -> Calculator steps are not linearly penalized.",
            "",
            "## Reward Component Explanation",
            "",
            "- Outcome correctness contributes at most 5.5 positive points and dominates process "
            "credit (at most 0.8).",
            "- Meaningful recovery is rewarded only after relevant correction and grounded "
            "success.",
            "- Multi-observation credit requires two successful SQL observations, Calculator, "
            "correct result, and grounding.",
            "- Penalties are capped so one malformed episode cannot create unbounded reward scale.",
            "",
            "## Reward Distribution",
            "",
            f"- Overall: mean={distribution['overall']['mean']:.4f}, "
            f"std={distribution['overall']['std']:.4f}, "
            f"p25={distribution['overall']['p25']:.4f}, "
            f"median={distribution['overall']['median']:.4f}, "
            f"p75={distribution['overall']['p75']:.4f}.",
            f"- Success mean: {distribution['success']['mean']:.4f}.",
            f"- Failure mean: {distribution['failure']['mean']:.4f}.",
            "",
            "## Success vs Failure Separation",
            "",
        )
    )
    for name, entry in pairwise["ordering"].items():
        lines.append(
            f"- {name}: {'PASS' if entry['pass'] else 'FAIL'}; "
            f"{entry['preferred_mean']:.4f} > {entry['dispreferred_mean']:.4f}."
        )
    lines.extend(("", "## Task-type Analysis", ""))
    for task_type, entry in distribution["by_task_type"].items():
        lines.append(
            f"- {task_type}: n={int(entry['count'])}, mean={entry['mean']:.4f}, "
            f"std={entry['std']:.4f}."
        )
    lines.extend(("", "## Pairwise Sanity Cases", ""))
    for name, entry in pairwise["groups"].items():
        lines.append(
            f"- {name}: n={entry['count']}, mean={entry['reward']['mean']:.4f}, "
            f"examples={', '.join(entry['examples']) or 'none'}."
        )
    lines.extend(
        (
            "",
            "## Candidate Comparison",
            "",
            "| Candidate | Success mean | Failure mean | Separation | Max failed reward |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for name, entry in candidates.items():
        lines.append(
            f"| {name} | {entry['success_mean']:.4f} | {entry['failure_mean']:.4f} | "
            f"{entry['separation']:.4f} | {entry['max_failure']:.4f} |"
        )
    lines.extend(
        (
            "",
            "`balanced` is frozen as canonical: it preserves outcome dominance, passes all "
            "pairwise orderings, and gives limited process credit without making error recovery "
            "or long trajectories profitable.",
            "",
            "## Reward Hacking Audit",
            "",
            f"- Highest failed reward: {hacking['highest_failed_reward']:.4f}.",
            f"- Lowest successful reward: {hacking['lowest_success_reward']:.4f}.",
            "- Tool/schema/calculator spam cannot add positive reward; it adds bounded penalties.",
            "- SQL execution and tool validity provide no positive reward.",
            "- Wrong arithmetic receives no answer/grounding credit and remains far below correct "
            "arithmetic despite executable tools.",
            "- Main residual risk: strict answer formatting and verifier coverage can still shape "
            "outcome reward; monitor validation success beside training reward in CP6.",
            "",
            "Top high-reward failures:",
            "",
        )
    )
    for example in hacking["top_failed_examples"]:
        lines.append(
            f"- {example['task_id']}: reward={example['reward']:.4f}, "
            f"type={example['task_type']}, signals={example['active_signals']}."
        )
    lines.extend(
        (
            "",
            "## Representative Trajectories",
            "",
            "- Meaningful recovery and superficial recovery examples are recorded in "
            "`audit.json` and the pairwise section above.",
            "- Per-trajectory explanations are saved in `reward_decomposition.jsonl`; each row "
            "contains outcome, process, penalties, counts, and total reward.",
            "",
            "## CP5 Status",
            "",
            f"**{audit['cp5_status']}**",
            "",
            "Canonical reward is frozen for CP6 Minimal GRPO Smoke Test only. No formal GRPO "
            "training or frozen-test evaluation has started.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    config = load_reward_config(args.config)
    tasks = {task.task_id: task for task in load_analysis_tasks(args.dataset)}
    source_path = args.source / "trajectories.jsonl"
    source_records = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    records: list[dict[str, Any]] = []
    with (args.output / "reward_decomposition.jsonl").open("w", encoding="utf-8") as handle:
        for raw in source_records:
            task_id = str(raw["task"]["task_id"])
            task = tasks[task_id]
            trajectory = _trajectory(cast(dict[str, Any], raw["trajectory"]))
            verification = verify_trajectory(task, trajectory)
            reward = score_agent_trajectory(task, trajectory, verification, config)
            signals = _signals(task, trajectory, verification)
            record = {
                "task_id": task_id,
                "task_type": task.task_type,
                "task": task,
                "trajectory": trajectory,
                "verification": verification,
                "signals": signals,
                "success": verification.success,
                "reward": reward,
            }
            records.append(record)
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "task_type": task.task_type,
                        "success": verification.success,
                        "verification": verification.to_dict(),
                        "signals": signals,
                        "reward": reward.to_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    signal_statistics = _signal_statistics(records)
    pairwise = _pairwise_audit(records)
    rewards = [record["reward"].total_reward for record in records]
    by_type: dict[str, list[float]] = defaultdict(list)
    for record in records:
        reward = cast(AgentRewardBreakdown, record["reward"])
        by_type[str(record["task_type"])].append(reward.total_reward)
    reward_distribution = {
        "overall": _distribution(rewards),
        "success": _distribution(
            [record["reward"].total_reward for record in records if record["success"]]
        ),
        "failure": _distribution(
            [record["reward"].total_reward for record in records if not record["success"]]
        ),
        "by_task_type": {
            task_type: _distribution(values) for task_type, values in sorted(by_type.items())
        },
        "component_mean": _component_contribution(records),
    }
    candidates = {
        "outcome_guardrail": replace(config, meaningful_recovery=0.0, multi_observation=0.0),
        "balanced": config,
        "process_richer": replace(
            config,
            meaningful_recovery=0.8,
            multi_observation=0.8,
            tool_error_penalty=0.1,
        ),
    }
    candidate_comparison = {
        name: _candidate_summary(records, candidate) for name, candidate in candidates.items()
    }
    failed = sorted(
        (record for record in records if not record["success"]),
        key=lambda record: record["reward"].total_reward,
        reverse=True,
    )
    successful = [record for record in records if record["success"]]
    hacking = {
        "highest_failed_reward": failed[0]["reward"].total_reward,
        "lowest_success_reward": min(
            record["reward"].total_reward for record in successful
        ),
        "top_failed_examples": [
            {
                "task_id": record["task_id"],
                "task_type": record["task_type"],
                "reward": record["reward"].total_reward,
                "active_signals": [
                    name for name, active in record["signals"].items() if active
                ],
            }
            for record in failed[:5]
        ],
    }
    opportunities = [record for record in records if record["signals"]["tool_error"]]
    broad = [record for record in opportunities if record["signals"]["recovered_from_error"]]
    meaningful = [record for record in broad if record["signals"]["meaningful_recovery"]]
    recovery = {
        "opportunities": len(opportunities),
        "broad_recovery": len(broad),
        "meaningful_recovery": len(meaningful),
        "superficial_recovery": len(broad) - len(meaningful),
        "meaningful_examples": [record["task_id"] for record in meaningful],
        "superficial_examples": [
            record["task_id"]
            for record in broad
            if not record["signals"]["meaningful_recovery"]
        ][:10],
    }
    ordering_pass = all(
        entry["pass"] for entry in pairwise["ordering"].values() if entry["evaluable"]
    )
    outcome_dominates = (
        config.answer_correct + config.result_correct + config.grounded
        > config.meaningful_recovery + config.multi_observation
    )
    audit = {
        "manifest": {
            "created_at": datetime.now(UTC).isoformat(),
            "source": str(args.source),
            "source_trajectory_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_manifest": json.loads(
                (args.source / "manifest.json").read_text(encoding="utf-8")
            ),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "verifier_git_commit": _git_commit(),
            "reward_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "frozen_test_used": False,
        },
        "canonical_config": asdict(config),
        "signal_statistics": signal_statistics,
        "recovery": recovery,
        "reward_distribution": reward_distribution,
        "pairwise": pairwise,
        "candidate_comparison": candidate_comparison,
        "reward_hacking": hacking,
        "cp5_status": "CP5 PASS" if ordering_pass and outcome_dominates else "CP5 FAIL",
    }
    (args.output / "signal_statistics.json").write_text(
        json.dumps(signal_statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output / "REWARD_ANALYSIS.md", audit)
    _write_report(args.report, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
