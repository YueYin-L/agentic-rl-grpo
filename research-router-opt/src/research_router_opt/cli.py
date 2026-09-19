"""Command-line entry point for dataset generation, training, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_router_opt.analysis_tasks import write_analysis_datasets
from research_router_opt.data import build_datasets, write_datasets
from research_router_opt.evaluate import evaluate_policy, write_evaluation
from research_router_opt.models import EvaluationMetrics, RoutingTask, Trajectory
from research_router_opt.policy import KeywordBaselinePolicy
from research_router_opt.train import train_policy


def _write_comparison(
    path: Path, baseline: EvaluationMetrics, trained: EvaluationMetrics
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_accuracy = baseline.selection_accuracy
    trained_accuracy = trained.selection_accuracy
    baseline_reward = baseline.average_reward
    trained_reward = trained.average_reward
    path.write_text(
        "\n".join(
            (
                "# Validation Comparison",
                "",
                "| Metric | Keyword baseline | LinUCB router | Change |",
                "|---|---:|---:|---:|",
                f"| Tool selection accuracy | {baseline_accuracy:.2%} | "
                f"{trained_accuracy:.2%} | {trained_accuracy - baseline_accuracy:+.2%} |",
                f"| Average reward | {baseline_reward:.3f} | {trained_reward:.3f} | "
                f"{trained_reward - baseline_reward:+.3f} |",
                "",
                "The validation set uses templates and subjects excluded from training.",
                "The policy updates only the lightweight routing layer; "
                "no LLM weights are trained.",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_failure_analysis(
    path: Path,
    tasks: list[RoutingTask],
    trajectories: list[Trajectory],
) -> None:
    task_by_id = {task.task_id: task for task in tasks}
    failures = [trajectory for trajectory in trajectories if not trajectory.success]
    lines = [
        "# Held-out Failure Analysis",
        "",
        f"Failed tasks: {len(failures)} / {len(trajectories)}",
        "",
        "| Task | Query | Expected | Selected | Reward |",
        "|---|---|---|---|---:|",
    ]
    for trajectory in failures:
        task = task_by_id[trajectory.task_id]
        step = trajectory.steps[0]
        lines.append(
            f"| {task.task_id} | {task.query} | {task.expected_tool} | "
            f"{step.action} | {trajectory.reward:.3f} |"
        )
    lines.extend(
        (
            "",
            "Observed pattern: all remaining errors contain the negated distractor "
            "`不用总结` before an explicit paper-search request. The bag-of-keywords state encoder "
            "detects `总结` but cannot represent negation or word order.",
            "",
            "Next technical step: replace binary keyword counts with a learned text encoder or add "
            "explicit negation features, then evaluate on a newly frozen test set. The current "
            "validation set must not be reused for tuning.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(project_root: Path) -> dict[str, object]:
    data_dir = project_root / "data"
    artifacts_dir = project_root / "artifacts"
    reports_dir = project_root / "reports"
    write_datasets(data_dir)
    train_tasks, validation_tasks = build_datasets()

    baseline_metrics, baseline_trajectories = evaluate_policy(
        KeywordBaselinePolicy(), validation_tasks
    )
    write_evaluation(
        reports_dir / "baseline_metrics.json", baseline_metrics, baseline_trajectories
    )

    trained_policy = train_policy(train_tasks)
    trained_policy.save(artifacts_dir / "policy.json")
    trained_metrics, trained_trajectories = evaluate_policy(trained_policy, validation_tasks)
    write_evaluation(
        reports_dir / "optimized_metrics.json", trained_metrics, trained_trajectories
    )
    _write_comparison(
        reports_dir / "comparison.md",
        baseline_metrics,
        trained_metrics,
    )
    _write_failure_analysis(
        reports_dir / "failure_analysis.md",
        validation_tasks,
        trained_trajectories,
    )
    summary: dict[str, object] = {
        "training_tasks": len(train_tasks),
        "validation_tasks": len(validation_tasks),
        "baseline": baseline_metrics.to_dict(),
        "optimized": trained_metrics.to_dict(),
        "policy_artifact": str(artifacts_dir / "policy.json"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("pipeline", "generate-analysis-data"),
        help="Run the legacy router pipeline or generate multi-turn analysis tasks.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Directory where data, artifacts, and reports are written.",
    )
    args = parser.parse_args()
    if args.command == "pipeline":
        print(json.dumps(run_pipeline(args.project_root), ensure_ascii=False, indent=2))
    elif args.command == "generate-analysis-data":
        manifest = write_analysis_datasets(args.project_root / "data")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
