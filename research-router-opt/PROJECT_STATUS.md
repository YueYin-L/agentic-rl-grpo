# Current Phase

GATE A — Checkpoint 1–3 complete; waiting for human review before the canonical real-model baseline.

# Sprint Day

DAY 1

# Latest Work

- Audited the existing one-step LinUCB router and preserved its verified pipeline.
- Added a model-independent multi-turn runtime with `MockBackend` and `VLLMBackend`.
- Added stateful DuckDB tools for schema inspection, read-only SQL, and safe arithmetic.
- Added deterministic train/validation/test task generation across seven task types.
- Added a semantic programmatic verifier and CPU-only regression tests.
- Generated and hashed 300 train, 150 validation, and 150 frozen test tasks.

# Checkpoint Status

CP1: PASS — Runtime, backend replacement, tool errors, timeout, reset, bounds, and serialization.

CP2: PASS — Stateful environment and 300/150/150 isolated task splits with taxonomy coverage.

CP3: PASS — Deterministic verifier for answer, SQL execution/result, tool validity, schema use,
grounding, recovery, step count, and redundant calls.

CP4: NOT STARTED — no canonical real-model baseline exists.

CP5: NOT STARTED — reward must be derived from baseline failures.

CP6: NOT STARTED — ART/GRPO/vLLM/LoRA are not installed or integrated.

CP7: NOT STARTED — no GRPO model checkpoint exists.

CP8: NOT STARTED — frozen test has not been opened for model evaluation.

# Tests

pytest: PASS — 19 tests.

Ruff: PASS.

MyPy: PASS — strict mode over the package.

# Compute

CPU: PASS for CP1–3.

Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 566.07,
CUDA driver capability 12.7.

Cloud GPU: not provisioned.

GPU currently required: NO for GATE A review; YES for the formal target-model baseline and GRPO.

# Dataset

train: 300 tasks, database fixture `train_analytics`.

validation: 150 tasks, database fixture `validation_analytics`.

test: 150 tasks, database fixture `test_analytics`; frozen and unused for tuning.

Hashes are recorded in `data/analysis_manifest.json`.

# Baseline

The legacy 24-task LinUCB result remains reproducible but is not the Agent GRPO baseline. No
Qwen/vLLM baseline claim is currently valid.

# Reward

No GRPO reward has been implemented. CP3 verifier signals are evidence inputs for CP5, not a
preselected reward formula.

# GRPO

Not started. No ART dependency, policy update, adapter, or training result is claimed.

# Evaluation

Verifier evaluation is CPU-tested. Canonical validation and frozen-test model evaluation are pending.

# Known Failures

- Exact target checkpoint is absent from repository configuration.
- Native Windows vLLM is unavailable in the current environment.
- Docker Desktop Linux engine is not running and WSL status access failed.

# Risks

1. The requested 9B-class model will not fit a stable BF16 vLLM baseline in 8 GiB local VRAM;
   checkpoint and quantization must be verified before selecting compute.
2. Synthetic templates may be too easy or expose wording artifacts; CP4 diagnostics must measure
   task-type difficulty before reward design.
3. The entire `research-router-opt/` directory is currently untracked from the parent detached HEAD,
   so reproducibility evidence is not protected by a commit yet.

# Deferred Work

SFT baseline, Docker Compose, W&B dashboards, UI, MCP, Redis, broad benchmark expansion, multiple
models, large hyperparameter searches, and extra reward ablations.

# Files Changed

- Added `analysis_models.py`, `backend.py`, `analysis_tools.py`, `environment.py`, `runtime.py`,
  `analysis_tasks.py`, `verifier.py`, and the CP4-ready `baseline.py` run writer.
- Added CP1–3 tests and generated `data/analysis_*.jsonl` plus the split manifest.
- Added DuckDB/pandas dependencies and kept the original LinUCB modules intact.

# Reproduction Commands

```powershell
$env:UV_CACHE_DIR=(Resolve-Path .uv-cache).Path
uv sync --extra dev
uv run research-router generate-analysis-data --project-root .
uv run pytest -q
uv run ruff check .
uv run mypy
```

# Git State

Parent checkout: detached `HEAD` at `41a44e13972c930e647be07dca31a2dc635fe468`.

Working tree: `research-router-opt/` is untracked in the parent repository. No commit was created.

# Next 1–3 Actions

1. GATE A review: confirm tasks require genuine multi-turn behavior and verifier semantics are sound.
2. Resolve an exact 9B-class checkpoint and a Linux OpenAI-compatible vLLM endpoint, then run a
   10–20-task validation smoke test.
3. Freeze the prompt/runtime configuration and run the 150-task canonical validation baseline.

## Checkpoint 4 Smoke Command

After `/v1/models` confirms the exact served ID, run:

```powershell
$env:VLLM_BASE_URL="http://<linux-gpu-host>:8000/v1"
$env:VLLM_MODEL="<exact-id-returned-by-v1-models>"
uv run python -m research_router_opt.baseline `
  --data data/analysis_validation.jsonl `
  --limit 20 `
  --max-steps 10 `
  --output results/diagnostics/cp4-smoke-001
```

The output directory must be new; the runner refuses to overwrite an existing formal run.
