# CP7 Formal GRPO Training Report

## Verdict

**CP7 PASS.** The controlled formal run completed all 100 policy updates and all five scheduled
150-task validation evaluations without touching the frozen test split. Checkpoint 80 is selected
for review because it has the highest validation Task Success and grounded-answer rate.

The final step-100 evaluation correctly triggered the frozen guard:

`validation reward increased (4.6630 -> 4.6903) while Task Success decreased (88.67% -> 87.33%)`

Consequently, step 100 is retained as the last checkpoint but is not promoted as the best model.
CP7 supplies validation evidence only; held-out improvement remains unproven until separately
authorized CP8 evaluation.

## Configuration

- Run ID: `cp7-formal-grpo-001`
- Training commit: `29f7206d25d5cad8589978bc355c03e7e553c9d8`
- Model: `Qwen/Qwen3.5-9B`
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Precision: BF16; no quantization
- Context length: 4096
- ART: 0.5.20 with `LocalBackend`
- vLLM runtime: 0.25.1 with the Blackwell FlashInfer sampler disabled
- LoRA: rank 8, alpha 16, dropout 0
- GRPO group size: 4
- Learning rate: `5e-6`
- Update budget: 100
- Checkpoint/validation cadence: every 20 updates
- Train tasks: frozen 300-task split
- Validation tasks: frozen 150-task split
- Frozen test loaded: **false**
- Reward: unchanged `configs/cp5_reward.toml`

Frozen hashes:

| Artifact | SHA-256 |
| --- | --- |
| CP5 reward | `ab225f57e747f4a3f74931c704a7f09a2bd5ab89acfa6b4703b03ab44ce9e99a` |
| Train | `d2334fe84e165519be152054e433a4966b8ea7862197a5c7d14f412d2a4a0292` |
| Validation | `95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c` |
| Frozen test | `34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b` |

## Training Timeline

| Checkpoint | Task Success | Final Accuracy | Result Correctness | Grounded | Recovery | Avg. Steps | Avg. Tools | Max-step | Reward mean/std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 49.33% | 54.00% | 55.33% | 49.33% | 91.43% | 5.81 | 4.93 | 11.33% | 1.2500 / 4.3504 |
| 20 | 72.67% | 73.33% | 77.33% | 72.67% | 96.97% | 5.07 | 4.12 | 5.33% | 3.2720 / 3.7230 |
| 40 | 69.33% | 70.67% | 74.67% | 69.33% | 100.00% | 4.50 | 3.52 | 2.00% | 3.1363 / 3.7049 |
| 60 | 74.67% | 76.00% | 80.00% | 74.67% | 83.33% | 4.25 | 3.25 | 0.00% | 3.6120 / 3.3575 |
| **80** | **88.67%** | **88.67%** | **91.33%** | **88.67%** | **100.00%** | 4.07 | 3.07 | 0.00% | 4.6630 / 2.4803 |
| 100 | 87.33% | 88.67% | 93.33% | 87.33% | 100.00% | **4.00** | **3.00** | 0.00% | 4.6903 / 2.2700 |

The non-monotonic checkpoints matter: step 40 regressed relative to step 20, step 80 became the
clear best validation policy, and step 100 improved result correctness and efficiency while losing
Task Success and grounding. Reward alone would have selected the wrong final checkpoint.

## Checkpoints

| Step | Role | Adapter SHA-256 |
| ---: | --- | --- |
| 20 | milestone | `a86e430ab68d91d5283c9b21156d230d515cbf592c974c0289c8938b204b6f24` |
| 40 | milestone | `e009d29f8a0f4bb39082b4ff0bbb99bf9bf55a518bec3e20a7e294228105ebd1` |
| 60 | milestone | `3a535598eb1445e4af225eaeafd660a67aabeb7fd9c0ff890d4e43e208a5de4e` |
| **80** | **best validation** | `7f4f843fbae65e9d9ffa2a0dfd0a79ec50e133fafa7131deef88454dd3fd372e` |
| 100 | last / guard-triggering | `990096831651a4c366f0dd90f69ef2153e61c81c6a1364be3746ac103ef09801` |

Complete step-80 and step-100 checkpoint copies are stored locally under the gitignored directory
`results/training/cp7-formal-grpo-001/checkpoints/`. Local hashes match the GPU host. Model binaries
are intentionally not committed to GitHub.

## Best Checkpoint Selection

The frozen ordering is: Task Success, grounded rate, result correctness, final accuracy, lower
invalid-call rate, then lower average steps. Under that rule, step 80 outranks step 100 despite the
latter's slightly higher reward and result correctness.

Step-80 validation by task type:

| Task type | CP4 Base | Step 80 | Delta |
| --- | ---: | ---: | ---: |
| Schema Discovery | 100.00% | 100.00% | +0.00 pp |
| Error Recovery | 86.36% | 95.45% | +9.09 pp |
| Multi-table Reasoning | 40.91% | 100.00% | +59.09 pp |
| SQL + Calculator | 4.76% | 95.24% | +90.48 pp |
| Multi-query Aggregation | 9.52% | 47.62% | +38.10 pp |
| Distractor Schema | 52.38% | 90.48% | +38.10 pp |
| Tool Selection | 47.62% | 90.48% | +42.86 pp |

Every task type is at least as strong as CP4 on validation at step 80. Multi-query remains the
weakest absolute category and is the primary residual target for failure analysis.

## Failure Analysis

At step 80, the remaining programmatic failures were:

- grounding failure: 17
- hallucination/wrong final: 17
- excessive tool use: 9
- SQL error: 7
- arithmetic error: 2
- tool-loop/max-step termination: 0
- invalid tool calls: 0
- potential shortcuts: 0

Compared with CP4, step 80 reduced grounding failures from 76 to 17, excessive tool use from 64
to 9, and eliminated max-step termination on validation. The remaining 17 grounding failures show
that GRPO did not solve reliable final-answer synthesis completely.

Step 100 regressed mainly in Error Recovery (95.45% -> 86.36%) and Multi-query Aggregation
(47.62% -> 38.10%) even though reward rose. This is the concrete failure case that justifies the
guard and best-checkpoint selection.

## Compute Statistics

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB
- Peak observed VRAM: 61,944 MiB
- Wall time: 31,701.8 s (8 h 48 min 22 s)
- Effective policy updates: 100
- Rollout-group attempts: 167
- Reward-degenerate groups skipped: 67
- Sum of effective-update rollout time: 6,157.5 s
- Sum of backend update time: 12,977.1 s
- Sum of five validation passes: 9,884.6 s
- OOM: false

The 67/167 degenerate-group rate is an important efficiency limitation: as behavior became more
consistent, many four-sample groups produced identical rewards and no usable relative GRPO signal.
This should inform future sampling/data design, but the frozen CP7 run was not altered mid-flight.

## Evidence

The lightweight canonical evidence is preserved under
`results/training/cp7-formal-grpo-001/`:

- `manifest.json`
- `training_state.json`
- generated `CP7_TRAINING_REPORT.md`
- validation `metrics.json` for steps 20/40/60/80/100

Complete training/validation trajectories and the best/last checkpoint are backed up locally in
the same gitignored run directory. No expected output, reward weight, dataset item, or frozen-test
result was changed during CP7.

## CP4 Comparison and Claim Boundary

On the frozen validation split, the selected step-80 checkpoint improved Task Success from 49.33%
to 88.67% (+39.33 percentage points), grounded-answer rate from 49.33% to 88.67%, and result
correctness from 55.33% to 91.33%, while reducing average tool calls from 4.93 to 3.07.

These are strong validation results, not held-out test results. The correct portfolio claim after
CP7 is that formal GRPO produced a checkpoint with substantially better frozen-validation behavior
and that a reward/Task-Success divergence was detected at the final checkpoint. Any held-out Base
vs GRPO claim must wait for the separately reviewed CP8 evaluation.
