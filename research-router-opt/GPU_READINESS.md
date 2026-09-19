# GPU Readiness

Status captured on 2026-09-19 after CP1–3.

## Target Model

- Requested class: Qwen3.5, approximately 9B parameters.
- Exact model/checkpoint: **UNRESOLVED**. No model identifier is present in repository config and
  `VLLM_MODEL` is unset. The code therefore does not hardcode a guessed checkpoint.
- Precision: unresolved until the checkpoint and serving method are verified.
- Context length: unresolved; runtime currently bounds behavior by `max_steps=10`.

## Local Compute

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
- VRAM: 8188 MiB total.
- Driver: 566.07.
- CUDA driver capability reported by `nvidia-smi`: 12.7.
- Current GPU use during audit: 0 MiB compute allocation.
- Native OS: Windows; vLLM is not installed.
- Docker Desktop Linux engine: unavailable/not running.
- WSL status: access failed in the current session.

## Local Model Cache

- Default Hugging Face hub cache exists and occupies approximately 1.52 GB.
- Only `Qwen/Qwen3-0.6B` was found in the cache.
- Target 9B-class model disk usage: unknown because the exact checkpoint is unresolved and absent.

## Software State

- Python: 3.11.7 in the project environment.
- `vllm`: not installed.
- `torch`: not installed in the project environment.
- ART: not installed.
- DuckDB: 1.5.5.
- pandas: 2.3.3.

## Runtime and Dataset Settings

- Canonical validation task count: 150 after a 10–20-task smoke test.
- Formal test task count: 150, frozen.
- Maximum agent steps: 10.
- Model timeout: 60 seconds per turn.
- Tool timeout: 5 seconds per call.
- Trajectory serialization: implemented and CPU-tested.
- Formal trajectory persistence/run IDs: implemented by the CP4 runner; output directory is the run ID.
- Resume support for baseline/training runs: not yet implemented.

## Readiness Decision

The local GPU is useful for a small-model or aggressively quantized compatibility diagnostic after
the serving stack is installed. It is not a credible default for stable BF16 vLLM inference or GRPO
training of an unresolved 9B-class model in 8 GiB VRAM.

Current classification for the requested target model: **CASE B — local GPU insufficient until proven
otherwise by a checkpoint-specific memory calculation and smoke test**.

Do not rent training compute yet. First resolve the exact checkpoint and run CP4 through a verified
Linux OpenAI-compatible vLLM endpoint. Use the smoke-test measurements to select the lowest stable
cloud GPU for GRPO.

## CP4 Preconditions

1. Record exact model ID and revision.
2. Record chat template/tool-call compatibility.
3. Provide a Linux vLLM endpoint and confirm `/v1/models` plus one tool-call request.
4. Persist the 10–20-task diagnostic separately from formal baseline results.
5. Freeze prompt, model, sampling, max steps, and timeout before the 150-task validation run.
