# Current Phase

DAY 2 — CP4 canonical baseline is BLOCKED BEFORE the 20-task real-model smoke.

# Sprint Day

DAY 2

# Latest Work

- Froze the completed CP1–3 implementation on a normal Git branch and commit.
- Locked the exact unquantized BF16 `Qwen/Qwen3.5-9B` checkpoint and tokenizer revision.
- Added a Linux vLLM 0.29.0 launcher using the official Qwen3.5 tool-call parser settings.
- Extended the existing baseline runner to enforce Git/dataset/model identity and persist
  trajectories, manifest, metrics, behavior checks, failures, compute statistics, and report.
- Kept Runtime, Environment, Verifier, tools, and all three dataset splits unchanged.
- Re-ran the full CPU quality gate: pytest, Ruff, and strict MyPy PASS.
- Diagnosed compute access: local 8 GiB GPU is not suitable for the canonical BF16 9B service;
  no authenticated Linux GPU vLLM endpoint is currently available.

# Checkpoint Status

CP1: PASS — unchanged.

CP2: PASS — unchanged; 300/150/150 isolated task splits remain frozen.

CP3: PASS — unchanged; deterministic verifier remains CPU-compatible.

CP4: BLOCKED BEFORE SMOKE — configuration and runner are ready, but 0/20 smoke and 0/150
canonical validation tasks have run because no accessible Linux GPU vLLM endpoint exists.

CP5: NOT STARTED — no reward components or weights are frozen without real baseline failures.

CP6: NOT STARTED.

CP7: NOT STARTED.

CP8: NOT STARTED — frozen test remains unused.

# Tests

pytest: PASS — 19 tests in 5.06 s.

Ruff: PASS.

MyPy: PASS — strict mode, 20 source files.

Linux launcher syntax: PASS via Git Bash `bash -n`.

TOML and baseline CLI load: PASS.

# Compute

CPU: PASS for CP1–3 and CP4 runner regression.

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 566.07.

Cloud GPU: not provisioned; remote SSH host was reachable but authentication was rejected.

GPU currently required: YES — Linux inference GPU for CP4 only, not GRPO training.

# Dataset

train: 300 tasks; unchanged.

validation: 150 tasks; SHA-256
`95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c`.

test: 150 tasks; frozen, unused; SHA-256
`34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b`.

# Baseline

Run ID: none — no real-model run was started.

Exact model: `Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

Precision: BF16, no quantized substitution. vLLM: 0.29.0 on Linux. Max model length:
8192. Chat template: checkpoint-bundled. Tool calling: OpenAI-compatible auto tool choice with
`qwen3_coder`, reasoning parser `qwen3`, thinking disabled, parallel tool calls disabled.

Key metrics: unavailable; 0/20 smoke and 0/150 canonical validation tasks executed.

# Reward

No reward has been selected. CP5 remains gated on the canonical baseline and failure analysis.

# GRPO

Not started. No ART/GRPO update, LoRA adapter, or training result is claimed.

# Evaluation

No real-model validation metric exists yet. Frozen test has not been used.

# Known Failures

- No authenticated Linux vLLM `/v1` endpoint is available.
- Real `/v1/models` identity, tool-call compatibility, OOM stability, and GPU utilization remain
  unverified.
- Agentic behavior and shortcuts cannot be assessed without real-model trajectories.

# Risks

1. A Linux GPU with insufficient headroom may OOM under the BF16 9B configuration; record peak VRAM
   during the 20-task smoke before raising concurrency.
2. Qwen may emit malformed tool calls or shortcut templated tasks; the smoke gate must classify this
   before the 150-task run.
3. CP5 reward design would be speculative if started before the canonical failure set exists.

# Deferred Work

SFT, formal GRPO, Docker Compose, W&B dashboards, UI, MCP, Redis, broad benchmarks, multi-model
comparison, hyperparameter search, and extra reward ablations.

# Files Changed

- `configs/cp4_qwen35_9b.toml`
- `scripts/serve_qwen35_9b.sh`
- `src/research_router_opt/backend.py`
- `src/research_router_opt/baseline.py`
- `tests/test_baseline.py`
- `BASELINE_REPORT.md`
- `PROJECT_STATUS.md`

# Reproduction Commands

Linux server:

```bash
bash scripts/serve_qwen35_9b.sh
```

20-task validation smoke from PowerShell:

```powershell
$env:VLLM_BASE_URL="http://<linux-gpu-host>:8000/v1"
$env:VLLM_API_KEY="EMPTY"
$env:BASELINE_GPU_NAME="<exact GPU name>"
$env:BASELINE_GPU_VRAM="<total VRAM>"
$env:BASELINE_GPU_DRIVER="<driver version>"
uv --cache-dir .uv-cache run python -m research_router_opt.baseline `
  --config configs/cp4_qwen35_9b.toml `
  --phase smoke `
  --limit 20 `
  --output results/diagnostics/cp4-smoke-001
```

150-task canonical validation, only after manual smoke review marks PASS:

```powershell
uv --cache-dir .uv-cache run python -m research_router_opt.baseline `
  --config configs/cp4_qwen35_9b.toml `
  --phase canonical `
  --limit 150 `
  --output results/baseline/cp4-qwen35-9b-validation-001
```

# Git State

Branch: `codex/agent-grpo-cp4-baseline`.

Gate A freeze commit: `ada7ec332cfc4655c9f5403f9c1d9f4787c3f877`.

CP4 runner/config commit: `e358e8c24ae592ca91cbde9331570913e4a09036`.

Working tree is expected to be clean after this status/report update is committed. Every actual
baseline manifest records the runtime commit and clean-tree state at execution time.

# Next 1–3 Actions

1. Provide or start an authenticated Linux GPU endpoint and run `scripts/serve_qwen35_9b.sh`.
2. Run and manually audit the 20-task validation smoke for engineering stability, genuine multi-turn
   behavior, difficulty, and shortcuts.
3. If and only if smoke passes, run all 150 validation tasks and hand the persisted trajectories to
   CP5 for CPU-only reward audit.
