# Canonical Agent Baseline Report

## Configuration

| Field | Canonical value |
|---|---|
| Run ID | `cp4-canonical-001` |
| Rollout commit | `670cdbda572697dae3690a275579a5b94c53bad0` |
| Audited verifier commit | `ddd656e2de9e8461f8c2db15dbc08328c8aa3f85` |
| Model | `Qwen/Qwen3.5-9B` |
| Model/tokenizer revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Precision | BF16, no quantization |
| Chat template | checkpoint-bundled `chat_template.jinja` |
| Tool calling | OpenAI-compatible auto tool choice; `qwen3_coder` parser |
| Thinking / parallel calls | disabled / disabled |
| vLLM | 0.29.0 on Linux |
| Context / output | 8192 / 1024 tokens |
| Agent limit | 10 steps; model timeout 120 s; tool timeout 5 s |
| Inference seed | `20260920` |
| Validation | 150 tasks; SHA-256 `95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c` |
| Frozen test | Not used; SHA-256 `34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b` |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 595.71.05 |

The original rollout artifacts are under `results/baseline/cp4-canonical-001/`. During
manual review, real trajectories exposed a deterministic verifier gap: valid agents may
derive a final scalar with Calculator instead of emitting the reference SQL's intermediate
aggregation. The fix did not change any model output. The persisted trajectories were
rescored offline under `results/baseline/cp4-canonical-001-audited/`, whose manifest records
both the original rollout commit and the verifier commit plus the source trajectory hash.

## Overall Metrics

| Metric | Audited result |
|---|---:|
| Task Success | 49.33% (74/150) |
| Final Answer Accuracy | 54.00% (81/150) |
| SQL Execution Success | 92.00% |
| Result Correctness | 55.33% |
| Tool Validity | 100.00% |
| Invalid Call Rate | 0.00% |
| Grounded Answer Rate | 49.33% |
| Recovery Rate | 91.43% (32/35 opportunities) |
| Average Steps | 5.81 |
| Average Tool Calls | 4.93 |
| Timeout Rate | 0.00% |

The raw pre-audit verifier reported 47.33% task success and 49.33% result correctness.
Offline semantic rescoring changed task success by only +2 percentage points, while
preserving all model outputs and compute measurements.

## Metrics by Task Type

| Task type | N | Task success | Final accuracy | Result correct | Avg. tool calls |
|---|---:|---:|---:|---:|---:|
| Schema Discovery | 22 | 100.00% | 100.00% | 100.00% | 2.00 |
| Error Recovery | 22 | 86.36% | 86.36% | 90.91% | 3.77 |
| Distractor Schema | 21 | 52.38% | 52.38% | 61.90% | 5.10 |
| Tool Selection | 21 | 47.62% | 47.62% | 47.62% | 5.48 |
| Multi-table Reasoning | 22 | 40.91% | 63.64% | 40.91% | 4.14 |
| Multi-query Aggregation | 21 | 9.52% | 19.05% | 38.10% | 8.19 |
| SQL + Calculator | 21 | 4.76% | 4.76% | 4.76% | 6.05 |

This is a non-degenerate baseline: the model reliably handles schema inspection but has
large, behavior-specific gaps on multi-observation aggregation and arithmetic semantics.

## Failure Modes

Programmatic counts can overlap for the same trajectory.

| Failure mode | Count |
|---|---:|
| Grounding failure | 76 |
| Excessive tool use | 64 |
| Hallucination / wrong final answer | 51 |
| Tool loop / max steps | 17 |
| Join/result error | 13 |
| Arithmetic tool error | 9 |
| SQL error | 5 |
| Wrong table | 2 |
| Wrong column | 2 |

The dominant SQL+Calculator error is semantic rather than infrastructural: the model often
uses profit divided by cost, while the task defines profit divided by discounted revenue.
Multi-query failures often begin with repeated guesses for nonexistent tables such as
`regions` or `order_details`, consuming the step budget before a final answer. Some otherwise
correct calculator outputs are rounded in the final answer beyond the frozen tolerance; those
remain failures rather than being silently relaxed after seeing validation results.

## Agentic Behavior Analysis

- Schema was called before SQL on 100% of Schema Discovery tasks.
- Every Multi-table task attempted a query with at least two joins.
- SQL + Calculator composition occurred on 92.86% of applicable tasks.
- Multi-query tasks obtained at least two SQL observations plus Calculator on 47.62%.
- 11.33% of all episodes terminated at the max-step limit.
- No premature final answer without tools was observed.
- No potential shortcut was found in the 150 audited trajectories.
- Tool-call schema validity was 100%; failures were primarily planning, relational semantics,
  arithmetic semantics, grounding, and efficiency failures.

## Representative Trajectories

### Correct schema-grounded answer

`validation-schema_discovery-0000` followed `schema -> sql -> final` and returned 48,
matching both the executed observation and expected answer.

### Successful error recovery

`validation-tool_selection-0006` first requested nonexistent table `customer`. It then used
the schema observations to locate `customers`, executed the corrected SQL, called Calculator,
and returned the correct uplifted value 543.45. This is a real error-observation-replan path.

### Valid multi-table composition

`validation-multi_table_reasoning-0001` inspected schema, joined `orders`, `customers`, and
`products`, retrieved discounted revenue rows, and used Calculator to obtain 3410.05. The
trajectory motivated the scalar-evidence verifier regression test.

### Multi-query loop

`validation-multi_query_aggregation-0004` repeatedly guessed nonexistent schema objects and
reached max_steps without SQL evidence or a final answer. This is a clear target for process
reward and efficiency penalties.

### Arithmetic-semantic failure

`validation-sql_calculator-0010` successfully queried data and used Calculator but computed
profit divided by cost instead of profit divided by discounted revenue, returning 103.59%
instead of 55.700182%. Tool execution success alone therefore cannot define reward.

## Candidate Reward Signals

These are CP5 candidates only; no weights are frozen here.

- Outcome: `answer_correct`, `result_correct`.
- Process: `tool_valid`, schema-grounded SQL, multi-observation completion,
  `recovered_from_error`, `grounded`.
- Penalties: calculator/SQL errors, repeated nonexistent-table probes, redundant calls,
  max-step termination, and excessive steps.
- Required audit: successful trajectories must score above failures; efficient success must
  score above looping success; strict arithmetic-semantic failures must not receive outcome
  credit merely because SQL executed.

## Compute Statistics

| Statistic | Value |
|---|---:|
| Total episode time | 2417.95 s (40.30 min) |
| Throughput | 3.72 tasks/min |
| Episode p50 | 12.98 s |
| Episode p95 | 32.27 s |
| Tool latency p50 | 5.55 ms |
| Tool latency p95 | 38.80 ms |

The vLLM server loaded the BF16 checkpoint without OOM. Model weights used about 16.8 GiB;
the running engine used about 27.5 GiB GPU memory during diagnostics.

## Reproduction

Run the frozen rollout configuration:

```powershell
$env:VLLM_BASE_URL="http://127.0.0.1:18000/v1"
$env:VLLM_API_KEY="EMPTY"
uv --cache-dir .uv-cache run python -m research_router_opt.baseline `
  --config configs/cp4_qwen35_9b.toml `
  --phase canonical --limit 150 `
  --output results/baseline/cp4-canonical-001
```

Re-run the deterministic verifier over the immutable trajectory file:

```powershell
uv --cache-dir .uv-cache run python scripts/rescore_baseline.py `
  --source results/baseline/cp4-canonical-001 `
  --output results/baseline/cp4-canonical-001-audited `
  --dataset data/analysis_validation.jsonl
```

## CP4 Status

**PASS.** The 20-task smoke and 150-task canonical validation baseline completed under the
frozen model, prompt, tool schema, inference configuration, dataset hash, and rollout commit.
Manual agentic-behavior and shortcut review passed. The frozen test split remains unused.

The persisted trajectories are sufficient for CPU-only CP5 reward design and sanity audit.
No formal GRPO training or held-out improvement is claimed at this checkpoint.
