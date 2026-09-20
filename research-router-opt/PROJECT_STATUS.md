# Current Phase

DAY 2 — CP4 canonical baseline PASS; ready for CP5 offline reward audit.

# Sprint Day

DAY 2

# Latest Work

- Served the exact pinned BF16 `Qwen/Qwen3.5-9B` checkpoint with Linux vLLM 0.29.0.
- Passed the 20-task validation smoke engineering and agentic-behavior gate.
- Completed the frozen 150-task canonical validation baseline and saved every trajectory.
- Manually audited schema use, error recovery, multi-table joins, SQL + Calculator,
  multi-observation behavior, loops, premature final answers, and shortcuts.
- Fixed two verifier gaps exposed by valid real-model decompositions and rescored the immutable
  trajectories offline with explicit rollout/verifier provenance.
- Did not use the frozen test split and did not start reward weighting or GRPO.

# Checkpoint Status

CP1: PASS — runtime unchanged.

CP2: PASS — environment and 300/150/150 isolated task splits unchanged.

CP3: PASS — deterministic CPU verifier; real-trajectory scalar evidence cases added to tests.

CP4: PASS — smoke `cp4-smoke-004`; canonical rollout `cp4-canonical-001`; audited output
`cp4-canonical-001-audited`.

CP5: READY — candidate signals identified; weights not frozen.

CP6: NOT STARTED.

CP7: NOT STARTED.

CP8: NOT STARTED — frozen test remains unused.

# Tests

pytest: PASS — 23 tests.

Ruff: PASS.

MyPy: PASS — 21 checked source/script files.

# Compute

CPU: PASS for runtime, environment, verifier, tests, aggregation, and offline rescoring.

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB; not used for canonical baseline.

Cloud GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 595.71.05.

GPU currently required: NO — CP5 reward audit runs over saved trajectories on CPU.

# Dataset

train: 300 tasks.

validation: 150 tasks; SHA-256
`95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c`.

test: 150 tasks; frozen and unused; SHA-256
`34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b`.

# Baseline

Run ID: `cp4-canonical-001`.

Exact model: `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, no quantization.

Rollout commit: `670cdbda572697dae3690a275579a5b94c53bad0`.

Audited verifier commit: `ddd656e2de9e8461f8c2db15dbc08328c8aa3f85`.

Core metrics: Task Success 49.33%; Final Accuracy 54.00%; SQL Execution 92.00%; Result
Correctness 55.33%; Tool Validity 100%; Invalid Calls 0%; Grounded Answers 49.33%; Recovery
91.43%; Average Steps 5.81; Average Tool Calls 4.93; Timeout 0%.

By type: Schema Discovery 100%; Error Recovery 86.36%; Distractor Schema 52.38%; Tool
Selection 47.62%; Multi-table 40.91%; Multi-query 9.52%; SQL + Calculator 4.76%.

Top failure modes: grounding failure 76; excessive tool use 64; hallucination/wrong final 51.

# Reward

No component weight is frozen. CP5 candidates come directly from baseline evidence:
answer/result correctness, tool validity, recovery, grounding, multi-observation completion,
and bounded efficiency penalties.

# GRPO

Not started. No ART/GRPO update, LoRA adapter, checkpoint, or improvement is claimed.

# Evaluation

Validation baseline only. Frozen test has not been opened for tuning or reporting.

# Known Failures

- SQL + Calculator frequently uses the wrong profit-rate denominator.
- Multi-query tasks often waste steps guessing nonexistent tables and terminate at max_steps.
- Correct tool outcomes are sometimes rounded or rewritten imprecisely in the final answer,
  causing strict grounding/accuracy failure.

# Risks

1. A reward dominated by SQL execution would reinforce wrong arithmetic semantics.
2. A reward dominated by final-answer matching could ignore loops and weak tool grounding.
3. Recovery rate alone is easy to overread because any later successful tool call is not
   necessarily a correct recovery; CP5 must inspect recovery trajectories explicitly.

# Deferred Work

SFT baseline, formal GRPO, Docker Compose, W&B dashboards, UI, MCP, Redis, multi-model search,
large hyperparameter search, and additional reward ablations.

# Files Changed

- `src/research_router_opt/backend.py`
- `src/research_router_opt/verifier.py`
- `src/research_router_opt/baseline.py`
- `scripts/rescore_baseline.py`
- `configs/cp4_qwen35_9b.toml`
- `tests/test_backend.py`
- `tests/test_verifier.py`
- `BASELINE_REPORT.md`
- `PROJECT_STATUS.md`

# Reproduction Commands

```powershell
uv --cache-dir .uv-cache run pytest -q
uv --cache-dir .uv-cache run ruff check .
uv --cache-dir .uv-cache run mypy src scripts/rescore_baseline.py
```

Canonical and offline-rescore commands are recorded in `BASELINE_REPORT.md`.

# Git State

Branch: `codex/agent-grpo-cp4-baseline`.

Gate A commit: `ada7ec332cfc4655c9f5403f9c1d9f4787c3f877`.

Canonical rollout commit: `670cdbda572697dae3690a275579a5b94c53bad0`.

Verifier audit commit: `ddd656e2de9e8461f8c2db15dbc08328c8aa3f85`.

Documentation update remains to be committed; working tree should be clean afterward.

# Next 1–3 Actions

1. Release the inference GPU; retain the model/checkpoint only if CP6 will start immediately.
2. Run CP5 reward decomposition over the saved audited trajectories on CPU.
3. Audit reward separation and hacking candidates before any GRPO smoke test.
