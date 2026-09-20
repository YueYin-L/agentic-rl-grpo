# Agent GRPO Implementation Plan

The existing repository is being extended, not replaced. The original CPU LinUCB router remains a
historical policy-layer experiment; the core path now reuses its typed traces, bounded execution,
evaluation discipline, CLI, tests, and evidence-first documentation.

## CORE

### DAY 1 — Environment and Verifier

- [x] Repository audit and legacy regression baseline.
- [x] CP1: model-independent multi-turn runtime.
- [x] CP2: stateful tool-use environment and isolated task splits.
- [x] CP3: deterministic programmatic verifier.
- [x] CPU pytest, Ruff, and strict MyPy.
- [x] Dataset manifest, task specification, and GPU readiness record.

Gate A requires human review of agentic task quality and verifier correctness.

### DAY 2 — Baseline, Failure Analysis, Reward, GRPO Smoke

- [x] Resolve the exact model checkpoint and canonical chat/tool-call template.
- [x] Run 10–20 validation diagnostics; do not publish these as formal metrics.
- [x] Freeze runtime/prompt settings and run the 150-task canonical validation baseline.
- [x] Produce task-type metrics and programmatic failure-mode analysis.
- [x] Design a small reward from observed failures and audit it offline on baseline trajectories.
- [ ] Integrate the installed ART API, vLLM rollout, and LoRA update using current official versions.
- [ ] Prove adapter parameter change, save/reload, and a post-update rollout.

Gate B requires human review of whether the observed failures are real, learnable, and correctly
represented by the proposed reward.

### DAY 3 — Formal GRPO and Evaluation

- [ ] Run one canonical GRPO configuration with recorded seed, revisions, hashes, and compute.
- [ ] Monitor validation task success together with reward to detect reward hacking.
- [ ] Freeze the trained adapter and evaluate Base vs GRPO once on the frozen 150-task test set.
- [ ] Run one failure-driven reward ablation.
- [ ] Preserve improvement, persistent-failure, and regression trajectories.
- [ ] Generate `PROJECT_SUMMARY.md` and `RESUME_EVIDENCE.md` from verified artifacts only.

Gate C determines what can be claimed in the portfolio and resume.

## NON-NEGOTIABLE

- Environment first, baseline first, reward second, RL third.
- Test is frozen until the final configuration is selected.
- All formal runs use run IDs and preserve config, metrics, trajectories, manifest, and compute data.
- A higher training reward is not evidence of better Agent behavior without held-out metrics.
- CPU tests remain independent from vLLM, CUDA, ART, and cloud services.

## DEFERRED

- SFT baseline unless compute/time remains after Base vs GRPO.
- Docker Compose until the GRPO loop is proven.
- Full W&B dashboard curation, UI, MCP, Redis, Kubernetes, and production service governance.
- Multiple models, large benchmark expansion, broad hyperparameter search, and multiple ablations.

## Stop Condition

The core project is complete only after a real adapter update and frozen-test Base-vs-GRPO report
exist with failure/regression analysis. After that point, new features do not enter the core path.
