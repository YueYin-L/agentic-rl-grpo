# Current Phase

DAY 3 — CP7 formal GRPO PASS; Gate C review required before CP8 frozen-test evaluation.

# Sprint Day

DAY 3

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
- Completed canonical CP6 run `cp6-smoke-015` on the exact Qwen3.5-9B revision with ART 0.5.20,
  vLLM 0.25.1, BF16, and LoRA r8.
- Proved non-degenerate reward groups, a real optimizer update, 256/256 changed LoRA tensors,
  checkpoint save, fresh-process step-1 reload, and a successful post-update tool-use trajectory.
- Preserved CP6 evidence locally and did not access the frozen test split or start formal training.
- Froze CP7 at 100 updates, group size 4, validation/checkpoint cadence 20, and the unchanged CP5
  reward/model/runtime/dataset protocol requested for the controlled formal run.
- Added deterministic train scheduling, non-degenerate-group gating, full 150-task validation at
  milestones, validation stop guards, checkpoint ranking, and an evidence-first report writer.
- Rechecked the frozen hashes and completed a successful CP6-equivalent preflight rollout on the
  96 GB Blackwell host.
- Synchronized commit `29f7206`, repeated the smoke without parser/OOM/runtime failure, and
  launched the formal run as background PID 17703 without loading the frozen test split.
- First effective update PASS: 4 trajectories, reward mean/std 5.425/0.075, rollout 48.1 s,
  update 179.7 s, non-zero loss/gradient/entropy, and approximately 61 GB peak VRAM.
- Completed 100 updates and five scheduled 150-task validation passes in 8 h 48 min.
- Selected checkpoint 80: 88.67% validation Task Success versus 49.33% CP4 Base, with every task
  type matching or exceeding its CP4 validation success rate.
- Correctly rejected checkpoint 100 after reward rose while Task Success fell; preserved both
  best and last checkpoints locally and did not access the frozen test split.

# Checkpoint Status

CP1: PASS — runtime unchanged.

CP2: PASS — environment and 300/150/150 isolated task splits unchanged.

CP3: PASS — deterministic CPU verifier; real-trajectory scalar evidence cases added to tests.

CP4: PASS — smoke `cp4-smoke-004`; canonical rollout `cp4-canonical-001`; audited output
`cp4-canonical-001-audited`.

CP5: PASS — canonical `configs/cp5_reward.toml` frozen after offline reward and hacking audit.

CP6: PASS — canonical run `cp6-smoke-015`; full rollout-to-reload engineering loop proven.

CP7: PASS — run `cp7-formal-grpo-001`; best checkpoint 80; final checkpoint 100 rejected by guard.

CP8: NOT STARTED — frozen test remains unused.

# Tests

pytest: PASS — 39 tests.

Ruff: PASS.

MyPy: PASS — 24 checked source/CPU-script files. The optional GPU-only
`scripts/run_cp6_smoke.py` is runtime-validated on Linux but excluded from the local MyPy gate
because ART, OpenAI, safetensors, and torch are intentionally absent from the CPU dev environment.

# Compute

CPU: PASS for runtime, environment, verifier, tests, aggregation, and offline rescoring.

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB; not used for canonical baseline.

Cloud GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, driver 580.82.09.

GPU currently required: NO — CP7 evidence and best/last checkpoints are backed up locally.

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

CP6 engineering smoke PASS. Run `cp6-smoke-015` used 3 train tasks x 4 generations, producing 12
trajectories with non-zero reward variance in every group. ART advanced from step 0 to step 1;
all 256 compared LoRA tensors changed (global delta L2 0.253665), and the step-1 adapter was saved.

A fresh process loaded step 1 and completed a six-step Schema -> SQL -> SQL -> SQL -> Calculator
trajectory with verifier success and reward 5.50. This proves pipeline integrity only; no policy
improvement is claimed before CP7/CP8 evaluation.

CP7 formal run `cp7-formal-grpo-001` completed 100 updates. Checkpoint 80 is selected with 88.67%
validation Task Success, 91.33% result correctness, 88.67% grounding, 4.07 average steps, and 3.07
average tool calls. Checkpoint 100 raised reward from 4.6630 to 4.6903 while Task Success declined
from 88.67% to 87.33%, so the frozen guard rejected it as best.

# Evaluation

Validation Base-vs-GRPO evidence exists through CP7. Frozen test remains unused; CP8 is not started.

# Known Failures

- SQL + Calculator frequently uses the wrong profit-rate denominator.
- Multi-query tasks often waste steps guessing nonexistent tables and terminate at max_steps.
- Correct tool outcomes are sometimes rounded or rewritten imprecisely in the final answer,
  causing strict grounding/accuracy failure.
- The original broad recovery rate was misleading: only 7/32 broad recoveries were meaningful.

# Risks

1. Strict answer formatting and verifier coverage can still shape outcome reward; CP7 must monitor
   task success and reward together.
2. The single validated correct SQL+Calculator baseline case makes arithmetic pairwise evidence
   directionally strong but sample-limited.
3. Process-richer weights slightly reduce the efficiency margin, so they remain rejected rather
   than being promoted because of marginally larger mean separation.
4. ART/vLLM/CUDA compatibility is version-sensitive: the validated Blackwell runtime requires an
   explicit CUDA runtime path and the FlashInfer sampler disabled.

# Deferred Work

SFT baseline, Docker Compose, W&B dashboards, UI, MCP, Redis, multi-model search, large
hyperparameter search, and additional reward ablations.

# Files Changed

- `src/research_router_opt/backend.py`
- `src/research_router_opt/verifier.py`
- `src/research_router_opt/baseline.py`
- `scripts/rescore_baseline.py`
- `scripts/analyze_rewards.py`
- `scripts/run_cp6_smoke.py`
- `scripts/run_cp7_training.py`
- `configs/cp4_qwen35_9b.toml`
- `configs/cp5_reward.toml`
- `configs/cp6_grpo_smoke.toml`
- `configs/cp7_grpo_formal.toml`
- `tests/test_backend.py`
- `tests/test_art_rollout.py`
- `tests/test_grpo_training.py`
- `tests/test_verifier.py`
- `tests/test_reward.py`
- `BASELINE_REPORT.md`
- `REWARD_ANALYSIS.md`
- `CP6_SMOKE_REPORT.md`
- `CP7_TRAINING_REPORT.md`
- `TRAINING_GPU_PLAN.md`
- `PROJECT_STATUS.md`

# Reproduction Commands

```powershell
uv --cache-dir .uv-cache run pytest -q
uv --cache-dir .uv-cache run ruff check .
uv --cache-dir .uv-cache run mypy src scripts/rescore_baseline.py scripts/analyze_rewards.py
```

Canonical and offline-rescore commands are recorded in `BASELINE_REPORT.md`.
CP6 Linux commands and evidence are recorded in `CP6_SMOKE_REPORT.md`.

# Git State

Branch: `codex/agent-grpo-cp4-baseline`.

Gate A commit: `ada7ec332cfc4655c9f5403f9c1d9f4787c3f877`.

Canonical rollout commit: `670cdbda572697dae3690a275579a5b94c53bad0`.

Verifier audit commit: `ddd656e2de9e8461f8c2db15dbc08328c8aa3f85`.

CP5 implementation commit: `ccd8430cbf9ae7a07a6bb4073bd2c74df78ea95c`.

CP5 audit-signal commit: `fde828b9532542fe35ff72250a591a4b13f2148a`.

CP6 runtime/config commit: `20987f6`.

CP6 report commit: `45000a6`.

CP7 frozen training implementation/run commit: `29f7206`.

# Next 1–3 Actions

1. Review CP7 checkpoint-80 selection, validation gains, residual failures, and guard behavior.
2. Freeze the exact CP8 Base-vs-GRPO evaluation command and checkpoint-80 adapter hash.
3. Only after review, run the frozen test once; do not retrain or tune on test results.
