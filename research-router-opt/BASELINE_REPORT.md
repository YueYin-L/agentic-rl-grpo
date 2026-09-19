# Canonical Agent Baseline Report

## Execution Status

CP4 is **BLOCKED BEFORE THE 20-TASK SMOKE TEST**. No real-model validation
trajectory was generated in this checkout, and the frozen test split was not used.

The blocking condition is compute access, not a CPU runtime, environment, verifier, or
dataset regression:

- the local NVIDIA GeForce RTX 4060 Laptop GPU has 8188 MiB VRAM and is not a stable
  target for the unquantized BF16 9B canonical configuration;
- native Windows vLLM is not the canonical Linux serving path;
- no authenticated Linux GPU vLLM endpoint or cloud instance credentials were available;
- an SSH probe reached `connect.westd.seetacloud.com` but authentication was rejected,
  so no remote process, model, or GPU was modified.

## Configuration

| Field | Frozen value |
|---|---|
| Branch | `codex/agent-grpo-cp4-baseline` |
| Gate A commit | `ada7ec332cfc4655c9f5403f9c1d9f4787c3f877` |
| CP4 runner/config commit | `e358e8c24ae592ca91cbde9331570913e4a09036` |
| Model ID | `Qwen/Qwen3.5-9B` |
| Model/tokenizer revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Precision | BF16; no GPTQ, AWQ, or 4-bit substitution |
| Chat template | checkpoint-bundled `chat_template.jinja` |
| Thinking | disabled for the canonical tool-use baseline |
| Tool calling | OpenAI Chat Completions, auto tool choice, non-parallel calls |
| vLLM | `0.29.0` on Linux |
| vLLM parsers | reasoning `qwen3`; tool calls `qwen3_coder` |
| Max model length | 8192 tokens |
| Max output tokens | 1024 tokens |
| Agent max steps | 10 |
| Model/tool timeouts | 120 s / 5 s |
| Validation hash | `95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c` |
| Frozen test hash | `34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b` |

The machine-readable configuration is `configs/cp4_qwen35_9b.toml`. The Linux
launcher is `scripts/serve_qwen35_9b.sh`. Before any task is run, the baseline CLI
checks that Git is clean, the validation hash is unchanged, and `/v1/models` contains
the exact configured model ID. Every completed run writes a manifest with Git state,
model and inference configuration, dataset hash, endpoint identity, task-ID hash, and
reported GPU metadata.

Official references used to lock the serving configuration:

- <https://huggingface.co/Qwen/Qwen3.5-9B>
- <https://huggingface.co/Qwen/Qwen3.5-9B/commit/c202236235762e1c871ad0ccb60c8ee5ba337b9a>
- <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>
- <https://docs.vllm.ai/en/latest/features/tool_calling/>
- <https://pypi.org/project/vllm/0.29.0/>

## Overall Metrics

Not available. Tasks executed: **0/150**.

No zero values are reported for success, accuracy, validity, recovery, grounding, or
timeout because zero would incorrectly imply a measured result rather than a run that
did not occur.

## Metrics by Task Type

Not available. No validation trajectory has been produced for any of the seven task
types.

## Failure Modes

Model behavior failure modes are not yet measurable. In particular, wrong table/column,
SQL/join/arithmetic errors, premature final answers, recovery failures, loops,
hallucinations, invalid calls, grounding failures, excessive tool use, and shortcuts
must not be inferred without real trajectories.

The three current execution blockers are:

1. no authenticated Linux GPU endpoint;
2. local 8 GiB VRAM is insufficient for the canonical unquantized BF16 9B service;
3. `/v1/models`, tool-call response compatibility, OOM behavior, and remote GPU
   statistics therefore remain unverified.

## Agentic Behavior Analysis

Not available. The 20-task smoke gate has not run, so there is no evidence yet for
schema-before-SQL behavior, error correction, cross-table joins, SQL/calculator
composition, multi-observation aggregation, premature final answers, looping, or
potential shortcuts.

## Representative Trajectories

None. No synthetic or MockBackend trajectory is substituted for the canonical Qwen
baseline.

## Candidate Reward Signals

No reward component or weight is frozen. CP5 must wait until the canonical baseline
produces a non-degenerate, auditable failure set. The runner can derive candidate
signals from verifier outputs and measured failure modes after CP4.

## Compute Statistics

| Resource | Status |
|---|---|
| CPU regression environment | PASS |
| Local GPU | RTX 4060 Laptop, 8188 MiB, driver 566.07 |
| Linux vLLM GPU | Not available |
| Cloud GPU | Not provisioned or authenticated |
| Real-model task time / throughput / peak VRAM | Not measured |

## Reproduction

On the Linux GPU host, from the repository's `research-router-opt` directory:

```bash
bash scripts/serve_qwen35_9b.sh
```

Verify the exact served identity from the client:

```bash
curl -s http://<linux-gpu-host>:8000/v1/models
```

From PowerShell in the same committed `research-router-opt` checkout, run the smoke:

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

Only after engineering and agentic-behavior review marks that smoke PASS, run:

```powershell
uv --cache-dir .uv-cache run python -m research_router_opt.baseline `
  --config configs/cp4_qwen35_9b.toml `
  --phase canonical `
  --limit 150 `
  --output results/baseline/cp4-qwen35-9b-validation-001
```

Each output directory is create-once and contains `manifest.json`, `metrics.json`,
`trajectories.jsonl`, and a generated `BASELINE_REPORT.md`.

## CP4 Status

**FAIL / BLOCKED BEFORE SMOKE.** This is not a model-quality failure and is not a CP5
handoff. CP4 becomes eligible for PASS only after the 20-task smoke and the full
150-task validation baseline both complete under the frozen configuration.

`CLOUD GPU CAN BE RELEASED` is **not applicable**: no cloud GPU was provisioned, and
the canonical baseline has not been saved.
