# Current Phase

DAY 2 — CP5 offline reward audit PASS; ready to prepare CP6 Minimal GRPO Smoke Test.

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
- Did not use the frozen test split and did not start GRPO.
- Audited verifier-signal activation, variance, overlap, conditional rates, and task-type coverage.
- Replaced broad recovery credit with a strict `meaningful_recovery` signal: 7 meaningful versus
  25 superficial recoveries among 32 broadly recovered trajectories.
- Compared three bounded reward candidates and froze the outcome-dominant balanced configuration.
- Persisted per-trajectory reward decompositions and passed all offline ordering/hacking checks.

# Checkpoint Status

CP1: PASS — runtime unchanged.

CP2: PASS — environment and 300/150/150 isolated task splits unchanged.

CP3: PASS — deterministic CPU verifier; real-trajectory scalar evidence cases added to tests.

CP4: PASS — smoke `cp4-smoke-004`; canonical rollout `cp4-canonical-001`; audited output
`cp4-canonical-001-audited`.

CP5: PASS — canonical `configs/cp5_reward.toml` frozen after offline reward and hacking audit.

CP6: READY, NOT STARTED — only Minimal GRPO Smoke Test preparation is authorized.

CP7: NOT STARTED.

CP8: NOT STARTED — frozen test remains unused.

# Tests

pytest: PASS — 30 tests.

Ruff: PASS.

MyPy: PASS — 22 checked source/script files.

# Compute

CPU: PASS for runtime, environment, verifier, tests, aggregation, and offline rescoring.

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB; not used for canonical baseline.

Cloud GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 595.71.05.

GPU currently required: NO — CP5 is complete; provision GPU only when CP6 smoke is ready to run.

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

CP5 run: `cp5-reward-audit-002` over immutable audited validation trajectories.

Canonical positive weights: answer correct 2.0; result correct 2.0; grounded 1.5; meaningful
recovery 0.4; valid multi-observation completion 0.4. Outcome positive mass (5.5) dominates
process positive mass (0.8).

Canonical penalties: wrong answer 1.0; wrong result 1.0; ungrounded 0.8; capped tool errors,
redundant calls, repeated nonexistent-object probes, excess calls, and max-step termination.

Reward distribution: overall mean 1.2500, std 4.3504, median 0.2000; success mean 5.4750;
failure mean -2.8638. Lowest success 5.05; highest failure 0.20.

All required pairwise audits PASS, including correct versus wrong arithmetic and meaningful versus
superficial recovery. `tool_valid` and `sql_exec_success` receive no positive reward.

# GRPO

Not started. CP6 may perform only a minimal ART/GRPO/LoRA save-reload smoke; no formal training,
checkpoint quality, or improvement is claimed.

# Evaluation

Validation baseline only. Frozen test has not been opened for tuning or reporting.

# Known Failures

- SQL + Calculator frequently uses the wrong profit-rate denominator.
- Multi-query tasks often waste steps guessing nonexistent tables and terminate at max_steps.
- Correct tool outcomes are sometimes rounded or rewritten imprecisely in the final answer,
  causing strict grounding/accuracy failure.
- The original broad recovery rate was misleading: only 7/32 broad recoveries were meaningful.

# Risks

1. Strict answer formatting and verifier coverage can still shape outcome reward; CP6 must monitor
   task success and reward together.
2. The single validated correct SQL+Calculator baseline case makes arithmetic pairwise evidence
   directionally strong but sample-limited.
3. Process-richer weights slightly reduce the efficiency margin, so they remain rejected rather
   than being promoted because of marginally larger mean separation.

# Deferred Work

SFT baseline, formal GRPO, Docker Compose, W&B dashboards, UI, MCP, Redis, multi-model search,
large hyperparameter search, and additional reward ablations.

# Files Changed

- `src/research_router_opt/backend.py`
- `src/research_router_opt/verifier.py`
- `src/research_router_opt/baseline.py`
- `scripts/rescore_baseline.py`
- `scripts/analyze_rewards.py`
- `configs/cp4_qwen35_9b.toml`
- `configs/cp5_reward.toml`
- `tests/test_backend.py`
- `tests/test_verifier.py`
- `tests/test_reward.py`
- `BASELINE_REPORT.md`
- `REWARD_ANALYSIS.md`
- `PROJECT_STATUS.md`

# Reproduction Commands

```powershell
uv --cache-dir .uv-cache run pytest -q
uv --cache-dir .uv-cache run ruff check .
uv --cache-dir .uv-cache run mypy src scripts/rescore_baseline.py scripts/analyze_rewards.py
```

Canonical and offline-rescore commands are recorded in `BASELINE_REPORT.md`.

# Git State

Branch: `codex/agent-grpo-cp4-baseline`.

Gate A commit: `ada7ec332cfc4655c9f5403f9c1d9f4787c3f877`.

Canonical rollout commit: `670cdbda572697dae3690a275579a5b94c53bad0`.

Verifier audit commit: `ddd656e2de9e8461f8c2db15dbc08328c8aa3f85`.

CP5 implementation commit: `ccd8430cbf9ae7a07a6bb4073bd2c74df78ea95c`.

CP5 audit-signal commit: `fde828b9532542fe35ff72250a591a4b13f2148a`.

CP5 report/status update remains to be committed; working tree should be clean afterward.

# Next 1–3 Actions

1. Inspect the currently installed ART API and freeze a minimal CP6 smoke configuration.
2. Select a few train tasks only, prove rollout -> reward -> LoRA update -> save -> reload.
3. Verify adapter parameters changed and a post-update rollout works; do not start formal GRPO.
