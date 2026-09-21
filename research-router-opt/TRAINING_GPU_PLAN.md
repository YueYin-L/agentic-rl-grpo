# Training GPU Plan

## Evidence Basis

The CP6 canonical smoke used one NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887 MiB
VRAM. Qwen3.5-9B BF16 inference, twelve ART rollouts, one LoRA/GRPO update, checkpoint save, and
fresh-process reload all completed.

- Training peak observed by `nvidia-smi`: 61,434 MiB
- Reload peak observed by `nvidia-smi`: 61,794 MiB
- Training OOM: no
- Earlier 32 GB GPU: inference worked, update OOMed
- Average trajectory length was not used as a sizing proxy; the actual packed update contained
  118,471 non-padding logical tokens and 14,853 loss-bearing assistant tokens.

## CP7 Recommendation

Use a single 80 GB-or-larger GPU for the same model, 4096 context, BF16, LoRA rank 8, group size 4,
and bounded concurrency. An 80 GB card is the lowest reasonable target inferred from the measured
approximately 62 GB peak plus operating margin; it has not yet been independently validated.

The already-provisioned 96 GB RTX PRO 6000 is a validated safe configuration and should be reused
if CP7 starts immediately. Do not increase context length, group size, or concurrent rollout count
at the same time as increasing the number of training steps.

## Frozen Starting Configuration

- Model: `Qwen/Qwen3.5-9B`
- Revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Precision: BF16, no quantization
- Context: 4096
- LoRA: r8 / alpha16
- GRPO group size: 4
- Trainer generations: 2
- Gradient accumulation sequences: 1
- Logprob chunk size: 8
- Learning rate: `5e-6`
- vLLM FlashInfer sampler: disabled on the validated Blackwell runtime
- CUDA runtime profile: cu130

## Formal-Run Controls

Before CP7, record one canonical budget instead of a grid search:

1. fixed train-task sample and seed;
2. update-step count and checkpoint cadence;
3. validation cadence and a reward-hacking stop condition;
4. rollout concurrency and timeout;
5. disk budget for adapters, trajectories, and logs;
6. resume test before committing to a long run.

If reward increases while validation task success, grounding, or invalid-call rate worsens, stop
and audit trajectories before spending more GPU time.

## Release Decision

CP6 evidence and the complete step-1 checkpoint have been copied off the GPU host. If CP7 will not
begin immediately, the cloud GPU can be released. CP5-style reward analysis and CP6 report
inspection are CPU-only.
