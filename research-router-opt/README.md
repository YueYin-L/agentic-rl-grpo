# Research Router Optimization → Agent GRPO Environment

This repository is being extended from a verified CPU-only research-routing experiment into a
multi-turn data-analysis environment for Agent GRPO post-training. The current verified core covers:

```text
ModelBackend → Multi-turn Runtime → Stateful Tools → Trajectory → Programmatic Verifier
```

The next evidence gate is the canonical real-model baseline. ART/GRPO, vLLM rollout, LoRA updates,
and Base-vs-GRPO metrics are **not yet claimed**. See `PROJECT_STATUS.md`, `TASK_SPEC.md`, and
`IMPLEMENTATION_PLAN.md` for the current boundary.

## Current multi-turn core

- A model-independent Python state machine supporting model → tool → observation → replan → final.
- Replaceable `MockBackend` and OpenAI-compatible `VLLMBackend` implementations.
- Stateful DuckDB schema, read-only SQL, and safe calculator tools with error feedback and timeouts.
- 300 train, 150 validation, and 150 frozen test tasks across seven multi-step behavior types.
- A deterministic semantic verifier for answers, SQL results, grounding, recovery, and redundancy.
- CPU-only tests for runtime, environment, tools, task generation, and verifier behavior.

## Preserved legacy experiment

The original modules remain a reproducible contextual-bandit experiment over the tool-routing policy
layer. They compare a fixed keyword baseline with LinUCB on 24 held-out synthetic routing tasks and
power the existing `/route` FastAPI demo. Those metrics are not evidence of LLM post-training.

## What this project is not

- It does not yet train or fine-tune LLM weights.
- It does not yet claim GRPO improvement, a trained LoRA adapter, or a formal real-model baseline.
- The local DuckDB fixtures are a controlled evaluation environment, not a production data service.

## Framework direction

The runtime and verifier remain framework-independent. The planned policy-update path is ART/GRPO
with vLLM rollout and LoRA after baseline failure analysis. No training framework is imported into
the CPU runtime or pytest path.

## Environment

- Python 3.11+
- `uv`
- No model download, GPU, or API key is required for CP1–3.

```powershell
uv sync --extra dev
uv run research-router generate-analysis-data --project-root .
uv run pytest -q
uv run ruff check .
uv run mypy

# Preserved legacy LinUCB experiment
uv run research-router pipeline --project-root .
```

Start the demo API after the pipeline has written `artifacts/policy.json`:

```powershell
uv run uvicorn research_router_opt.api:app --host 127.0.0.1 --port 8000
```

```http
POST /route
Content-Type: application/json

{"query": "总结这篇亚洲象论文的摘要"}
```

## Legacy LinUCB data and evaluation

The pipeline deterministically generates:

- 48 training tasks from training-only templates and subjects;
- 24 validation tasks from disjoint templates and subjects;
- four actions: `species_lookup`, `paper_search`, `evidence_compare`, and
  `abstract_summarize`.

The transparent reward is:

```text
+0.65 correct tool / -0.35 wrong tool
+0.20 successful execution
+0.20 valid output schema
-0.05 per tool call
```

The verified metrics are written to:

- `reports/baseline_metrics.json`
- `reports/optimized_metrics.json`
- `reports/comparison.md`
- `reports/failure_analysis.md`

Latency covers only local fixture execution and must not be presented as model or network latency.

### Verified result

On 24 held-out synthetic tasks whose templates and subjects do not occur in training:

| Metric | Keyword baseline | LinUCB router | Change |
|---|---:|---:|---:|
| Tool selection accuracy | 50.0% | 87.5% | +37.5 percentage points |
| Average reward | 0.40 | 0.85 | +0.45 |

All three remaining failures share the same negated-distractor form: `不用总结，搜索……`.
The keyword-count state encoder represents the word `总结` but not its negation or order. See
`reports/failure_analysis.md`; do not tune against the frozen validation set.

## Project structure

```text
src/research_router_opt/
  analysis_models.py  multi-turn task, state, trace, and verifier models
  backend.py          Mock and OpenAI-compatible vLLM backends
  runtime.py          bounded multi-turn state machine
  environment.py      stateful episode lifecycle
  analysis_tools.py   DuckDB schema/SQL/calculator tools
  analysis_tasks.py   deterministic 300/150/150 split generation
  verifier.py         semantic programmatic trajectory verifier
  baseline.py         persistent real-model validation runner
  agent.py       bounded agent loop
  api.py         FastAPI demo
  cli.py         reproducible pipeline
  data.py        deterministic train/validation generation
  evaluate.py    metrics and trajectory export
  features.py    state encoder
  models.py      typed task/trace/result models
  policy.py      keyword baseline and LinUCB policy
  reward.py      auditable reward components
  tools.py       deterministic local tools
  train.py       offline policy-update loop
tests/           legacy and multi-turn CPU regression tests
```

## Resume boundary

Safe current description: "implemented a multi-turn tool-use environment and deterministic
trajectory verifier, with a preserved LinUCB routing baseline." Unsafe description: "fine-tuned an
LLM with GRPO." Only results produced by completed checkpoints may be quoted.

The ready-to-review Chinese resume draft and interview boundary are in
`docs/resume_project_cn.md`.
