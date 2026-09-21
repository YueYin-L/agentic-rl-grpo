# CP6 Minimal GRPO Smoke Report

## Verdict

**CP6 PASS** for the engineering objective only.

The run proved the complete path:

`Qwen3.5-9B -> ART rollout -> trajectory -> verifier -> frozen CP5 reward -> GRPO/PPO update -> LoRA checkpoint -> fresh-process reload -> post-update rollout`

This run does **not** establish policy-quality improvement. It used three train tasks, twelve
pre-update trajectories, and one optimizer step. Improvement remains a CP7/CP8 question requiring
validation monitoring and the untouched frozen test split.

## Canonical Run

- Run ID: `cp6-smoke-015`
- Git commit used for the run: `20987f6`
- Model: `Qwen/Qwen3.5-9B`
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Precision: BF16; no quantization
- Context length: 4096
- Tool parser: `qwen3_xml`
- ART: 0.5.20 with `LocalBackend`
- vLLM runtime: 0.25.1, CUDA 13 profile
- LoRA: rank 8, alpha 16, dropout 0
- Learning rate: `5e-6`
- GRPO groups: 3 tasks x 4 generations = 12 trajectories
- Update count: 1
- Frozen reward: `configs/cp5_reward.toml`
- Train dataset SHA-256:
  `d2334fe84e165519be152054e433a4966b8ea7862197a5c7d14f412d2a4a0292`
- Frozen test use: none

The exact run configuration is in `configs/cp6_grpo_smoke.toml`.

## Rollout and Reward Gate

All three task groups had non-zero within-group reward variance, so GRPO had a usable relative
learning signal:

| Task | Rewards | Variance | Gate |
| --- | --- | ---: | --- |
| SQL + Calculator | 5.45, -2.80, -2.80, -5.40 | 16.7105 | PASS |
| Multi-query Aggregation | -3.80, -5.15, -4.40, 5.75 | 19.7363 | PASS |
| Multi-table Reasoning | 5.75, -2.95, 5.35, -2.80 | 17.7680 | PASS |

Four of twelve pre-update trajectories succeeded. The failed samples included wrong answers,
wrong results, ungrounded answers, max-step loops, and redundant calls. This mixture confirms that
the smoke was neither all-success nor all-failure and that the frozen reward separated behaviors
inside each task group.

## Update Evidence

- Initial ART step: 0
- Result ART step: 1
- Trainable assistant tokens: 14,853
- Packed sequences: 82
- Mean reported training loss: 0.2802
- Mean gradient norm: 1.3720
- Mean entropy: 0.3950
- Adapter-before SHA-256:
  `2f96cc1bd277c2921732baeff5ad5c322504f14fb2a41b93ee5d007efa0f5fcb`
- Adapter-after SHA-256:
  `233a0c5ea68f872cddce889da36481cb6b21ee590f623972cce285e5e5906b7c`
- Common LoRA tensors compared: 256
- Changed tensors: 256
- Global delta L2: 0.253665
- Maximum absolute tensor delta: 0.001588
- Saved adapter size: 58,233,376 bytes

Loss alone was not accepted as evidence. The changed hashes and tensor-level delta prove that the
LoRA weights were actually updated.

## Fresh-Process Reload

A new SSH/Python process registered the saved model at ART step 1 and generated a new trajectory.

- Process-boundary flag: true
- Loaded step: 1
- Task: `train-sql_calculator-0003`
- Termination: final answer
- Steps: 6
- Tool sequence: Schema -> SQL -> SQL -> SQL -> Calculator
- Verifier success: true
- Answer correct: true
- Result correct: true
- Grounded: true
- Tool valid: true
- Redundant calls: 0
- Reward: 5.50

This proves that the saved adapter can be loaded and used for a later rollout. It is a functional
check, not a statistically meaningful post-training comparison.

## Compute

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- VRAM: 97,887 MiB reported by `nvidia-smi`
- Driver: 580.82.09
- CUDA driver capability: 13.0
- Rollout time: 244.0 s
- Training backend time: 612.0 s
- End-to-end training phase time: 1,031.4 s
- Fresh-process reload and rollout: 121.2 s
- Post-update trajectory generation: 23.4 s
- Training peak GPU memory: 61,434 MiB
- Reload peak GPU memory: 61,794 MiB
- OOM in canonical run: false

## Compatibility Findings

The smoke exposed four integration details that matter for production Agentic RL:

1. ART training requires the exact prompt/completion token IDs and rollout logprobs from vLLM;
   text-only trajectory capture is insufficient.
2. Qwen3 XML tool-call parsing is part of the policy/runtime interface. A parser mismatch can make
   valid model actions appear degenerate.
3. ART 0.5.20 deadlocked with the attempted accumulation queue above one sequence; the smoke uses
   `gradient_accumulation_sequences = 1`.
4. On this Blackwell runtime, FlashInfer 0.6.13 misidentified `sm_120`; disabling only the
   FlashInfer sampler allowed vLLM's PyTorch sampler to run. The runtime also needs an explicit
   CUDA runtime library path.

Earlier 32 GB runs completed inference but OOMed during the update. Those failed attempts were not
promoted into the canonical result.

## Evidence Files

The canonical artifacts are preserved under `results/training/cp6-smoke-015/`:

- `train_evidence.json`
- `reload_evidence.json`
- `rollout_audit.json`
- `pre_update_trajectories.jsonl`

The ART checkpoint is stored on the training host at:

`/root/autodl-tmp/art-cp6/agentic-rl-grpo/models/cp6-minimal-grpo-smoke/checkpoints/0001`

## CP7 Handoff

CP7 may now prepare one canonical formal GRPO configuration. Before training, freeze the task
sample, total update budget, evaluation cadence, seed, checkpoint cadence, and stop rule. Monitor
validation task success, invalid calls, grounding, recovery, average steps, and reward together.
Do not touch the frozen test split until the final adapter and Base-vs-GRPO evaluation configuration
are frozen.
