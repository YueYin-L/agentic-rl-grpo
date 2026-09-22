# CP7 Formal GRPO Training Report

## Status

- status: stopped_by_validation_guard
- completed update steps: 100
- best validation step: 80
- stop reasons: validation_reward_up_but_task_success_down

No frozen-test task was loaded or evaluated during CP7.

## Configuration

```json
{
  "run_id": "cp7-formal-grpo-001",
  "started_at": "2026-09-21T09:02:47.919444+00:00",
  "git": {
    "branch": "main",
    "commit": "29f7206d25d5cad8589978bc355c03e7e553c9d8",
    "tracked_working_tree_clean": true
  },
  "frozen_hashes": {
    "reward": "ab225f57e747f4a3f74931c704a7f09a2bd5ab89acfa6b4703b03ab44ce9e99a",
    "train": "d2334fe84e165519be152054e433a4966b8ea7862197a5c7d14f412d2a4a0292",
    "validation": "95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c",
    "test": "34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b"
  },
  "config": {
    "model": {
      "id": "Qwen/Qwen3.5-9B",
      "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
      "precision": "bfloat16",
      "quantization": "none",
      "max_seq_length": 4096,
      "tool_call_parser": "qwen3_xml"
    },
    "reward": {
      "config": "configs/cp5_reward.toml",
      "sha256": "ab225f57e747f4a3f74931c704a7f09a2bd5ab89acfa6b4703b03ab44ce9e99a"
    },
    "dataset": {
      "train_path": "data/analysis_train.jsonl",
      "train_sha256": "d2334fe84e165519be152054e433a4966b8ea7862197a5c7d14f412d2a4a0292",
      "validation_path": "data/analysis_validation.jsonl",
      "validation_sha256": "95baa5c0438d455a7063ded7679e714d29807b0b575971eee46c692c02f1d98c",
      "test_path": "data/analysis_test.jsonl",
      "test_sha256": "34b6d73f32a062c4a5e7fb1bee124aa402ebb8f37f1052b450fabda3bfa0765b"
    },
    "runtime": {
      "max_steps": 10,
      "model_timeout_s": 180.0,
      "tool_timeout_s": 5.0
    },
    "rollout": {
      "group_size": 4,
      "temperature": 0.8,
      "top_p": 0.95,
      "max_output_tokens": 768
    },
    "validation": {
      "every_steps": 20,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_output_tokens": 1024,
      "seed": 20260920
    },
    "training": {
      "max_update_steps": 100,
      "learning_rate": 5e-06,
      "loss_fn": "ppo",
      "scale_rewards": true,
      "gradient_accumulation_sequences": 1,
      "trainer_num_generations": 2,
      "logprob_calculation_chunk_size": 8,
      "seed": 20260920,
      "max_group_attempts": 600
    },
    "checkpoint": {
      "save_every_steps": 20,
      "keep_last": true,
      "keep_best_validation": true
    },
    "baseline": {
      "run_id": "cp4-canonical-001-audited",
      "task_success": 0.49333333333333335,
      "final_answer_accuracy": 0.54,
      "result_correctness": 0.5533333333333333,
      "grounded_answer_rate": 0.49333333333333335,
      "recovery_rate": 0.9142857142857143,
      "average_steps": 5.8133333333333335,
      "average_tool_calls": 4.926666666666667,
      "invalid_call_rate": 0.0,
      "max_step_termination_rate": 0.11333333333333333,
      "reward_mean": 1.25
    },
    "lora": {
      "rank": 8,
      "alpha": 16,
      "dropout": 0.0,
      "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
      ]
    },
    "art": {
      "version": "0.5.20",
      "backend": "LocalBackend",
      "path": "/root/autodl-tmp/art-cp7",
      "project": "agentic-rl-grpo",
      "model_name": "qwen35-9b-cp7-formal",
      "run_name": "cp7-formal-grpo-001"
    },
    "compute": {
      "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      "vram_mib": 97887,
      "cuda_profile": "cu130"
    }
  },
  "train_task_count": 300,
  "validation_task_count": 150,
  "test_loaded": false,
  "flashinfer_sampler_disabled": true
}
```

## Training Timeline

| Step | Task | Reward mean | Reward std | Rollout s | Train s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | train-distractor_schema-0236 | 5.4250 | 0.0750 | 48.1 | 179.7 |
| 2 | train-multi_table_reasoning-0120 | 2.7500 | 4.3882 | 82.2 | 184.2 |
| 3 | train-multi_table_reasoning-0246 | 5.4250 | 0.0750 | 85.3 | 133.0 |
| 4 | train-multi_table_reasoning-0148 | 3.2000 | 3.7284 | 69.1 | 164.3 |
| 5 | train-sql_calculator-0115 | -0.7625 | 3.6162 | 75.1 | 133.0 |
| 6 | train-tool_selection-0160 | 1.6250 | 3.9213 | 62.5 | 157.7 |
| 7 | train-sql_calculator-0136 | 2.7750 | 4.3803 | 72.1 | 179.1 |
| 8 | train-distractor_schema-0117 | 5.3625 | 0.1474 | 61.3 | 164.2 |
| 9 | train-distractor_schema-0040 | 5.4250 | 0.0750 | 49.9 | 128.2 |
| 10 | train-error_recovery-0051 | 5.4625 | 0.0650 | 50.9 | 108.6 |
| 11 | train-error_recovery-0191 | 5.4875 | 0.0217 | 42.1 | 108.6 |
| 12 | train-distractor_schema-0012 | 3.4250 | 3.5940 | 56.0 | 118.7 |
| 13 | train-tool_selection-0041 | 1.3500 | 4.1500 | 50.4 | 108.3 |
| 14 | train-multi_query_aggregation-0144 | -3.1750 | 1.9185 | 75.6 | 204.2 |
| 15 | train-distractor_schema-0166 | 5.4625 | 0.0650 | 54.5 | 123.5 |
| 16 | train-error_recovery-0135 | 5.5125 | 0.2859 | 60.6 | 123.4 |
| 17 | train-multi_query_aggregation-0235 | -0.6625 | 1.3221 | 86.1 | 164.3 |
| 18 | train-sql_calculator-0213 | -0.7625 | 3.6162 | 79.2 | 123.9 |
| 19 | train-sql_calculator-0297 | 1.2250 | 4.0289 | 126.3 | 138.3 |
| 20 | train-sql_calculator-0248 | 3.3375 | 3.5437 | 76.1 | 147.7 |
| 21 | train-distractor_schema-0033 | 4.1000 | 2.2550 | 44.4 | 114.0 |
| 22 | train-tool_selection-0265 | 1.3500 | 4.1500 | 52.6 | 118.5 |
| 23 | train-multi_query_aggregation-0081 | 4.0250 | 2.6515 | 95.8 | 184.6 |
| 24 | train-sql_calculator-0080 | 5.2000 | 0.2806 | 115.6 | 123.8 |
| 25 | train-multi_query_aggregation-0200 | 3.0125 | 3.5303 | 65.1 | 143.6 |
| 26 | train-multi_query_aggregation-0053 | 5.7875 | 0.1949 | 72.8 | 164.5 |
| 27 | train-distractor_schema-0180 | 1.2000 | 4.2283 | 57.9 | 133.2 |
| 28 | train-multi_table_reasoning-0001 | 3.4875 | 3.6315 | 43.8 | 87.3 |
| 29 | train-tool_selection-0062 | 3.4875 | 3.6315 | 38.7 | 113.8 |
| 30 | train-tool_selection-0076 | 3.4250 | 3.5940 | 46.1 | 118.4 |
| 31 | train-multi_query_aggregation-0151 | 1.0875 | 4.8176 | 82.5 | 179.2 |
| 32 | train-multi_query_aggregation-0291 | 3.2375 | 4.5828 | 72.9 | 174.3 |
| 33 | train-distractor_schema-0131 | 1.3500 | 4.1500 | 35.5 | 82.3 |
| 34 | train-tool_selection-0195 | 5.4375 | 0.2103 | 57.4 | 152.6 |
| 35 | train-tool_selection-0153 | 3.4250 | 3.5940 | 51.9 | 128.2 |
| 36 | train-sql_calculator-0059 | 1.9500 | 3.6500 | 89.4 | 138.2 |
| 37 | train-sql_calculator-0031 | 1.3125 | 4.1878 | 69.7 | 143.2 |
| 38 | train-multi_query_aggregation-0067 | 5.8625 | 0.2328 | 74.7 | 138.1 |
| 39 | train-tool_selection-0279 | 1.3500 | 4.1500 | 31.3 | 86.8 |
| 40 | train-error_recovery-0282 | 5.5625 | 0.1083 | 47.9 | 91.9 |
| 41 | train-sql_calculator-0129 | 1.9875 | 3.5978 | 84.6 | 143.2 |
| 42 | train-tool_selection-0188 | 1.3500 | 4.1500 | 44.4 | 118.4 |
| 43 | train-tool_selection-0097 | 1.2625 | 4.2393 | 46.6 | 123.2 |
| 44 | train-multi_table_reasoning-0274 | 5.4625 | 0.0650 | 52.1 | 103.8 |
| 45 | train-multi_table_reasoning-0106 | 5.4250 | 0.1299 | 59.2 | 103.2 |
| 46 | train-tool_selection-0132 | 1.9500 | 3.6500 | 59.1 | 118.5 |
| 47 | train-multi_query_aggregation-0277 | -2.6250 | 1.8386 | 59.9 | 157.8 |
| 48 | train-multi_query_aggregation-0242 | -2.0875 | 1.3221 | 69.6 | 138.3 |
| 49 | train-multi_query_aggregation-0088 | 0.8750 | 3.1491 | 73.8 | 133.6 |
| 50 | train-sql_calculator-0101 | 3.3000 | 3.5240 | 103.2 | 143.0 |
| 51 | train-error_recovery-0058 | 5.5625 | 0.1083 | 40.0 | 92.0 |
| 52 | train-multi_query_aggregation-0256 | 1.7125 | 4.1750 | 79.0 | 157.9 |
| 53 | train-multi_table_reasoning-0260 | 5.4250 | 0.1299 | 51.8 | 108.7 |
| 54 | train-multi_query_aggregation-0186 | 5.6750 | 0.1601 | 75.8 | 158.0 |
| 55 | train-sql_calculator-0185 | 1.3625 | 4.1655 | 109.0 | 129.0 |
| 56 | train-tool_selection-0048 | -3.5625 | 0.8488 | 59.8 | 181.4 |
| 57 | train-tool_selection-0104 | 3.4250 | 3.5940 | 34.8 | 105.0 |
| 58 | train-multi_table_reasoning-0204 | 5.4625 | 0.0650 | 75.1 | 124.6 |
| 59 | train-tool_selection-0293 | 1.3500 | 4.1500 | 43.0 | 109.5 |
| 60 | train-sql_calculator-0087 | -1.1000 | 3.8277 | 83.4 | 143.5 |
| 61 | train-error_recovery-0037 | 5.5625 | 0.1083 | 40.5 | 87.1 |
| 62 | train-error_recovery-0072 | 5.5250 | 0.0433 | 59.6 | 113.2 |
| 63 | train-distractor_schema-0299 | 5.4250 | 0.1299 | 56.3 | 123.1 |
| 64 | train-tool_selection-0202 | 0.8750 | 4.5922 | 61.4 | 147.2 |
| 65 | train-schema_discovery-0042 | 3.4250 | 3.5940 | 29.3 | 82.1 |
| 66 | train-tool_selection-0167 | 3.4250 | 3.5940 | 41.9 | 103.2 |
| 67 | train-sql_calculator-0262 | 1.7000 | 3.8321 | 81.0 | 144.2 |
| 68 | train-multi_query_aggregation-0060 | 2.3250 | 3.7796 | 73.9 | 143.1 |
| 69 | train-tool_selection-0174 | 3.4000 | 3.5859 | 44.4 | 133.3 |
| 70 | train-sql_calculator-0269 | -0.7750 | 3.6826 | 81.1 | 128.5 |
| 71 | train-tool_selection-0118 | -3.2375 | 0.6740 | 60.0 | 138.6 |
| 72 | train-multi_query_aggregation-0130 | -0.6250 | 1.3437 | 93.3 | 153.8 |
| 73 | train-multi_table_reasoning-0022 | 5.4625 | 0.0650 | 52.0 | 99.5 |
| 74 | train-tool_selection-0034 | 3.4250 | 3.5940 | 42.0 | 110.3 |
| 75 | train-tool_selection-0055 | 0.9875 | 4.5415 | 49.4 | 135.9 |
| 76 | train-tool_selection-0216 | 3.3500 | 3.7239 | 55.4 | 125.4 |
| 77 | train-multi_table_reasoning-0169 | 3.4250 | 3.5940 | 36.1 | 82.7 |
| 78 | train-multi_query_aggregation-0102 | 1.5000 | 2.5485 | 64.4 | 124.2 |
| 79 | train-distractor_schema-0285 | 3.4250 | 3.5940 | 45.3 | 104.6 |
| 80 | train-multi_query_aggregation-0109 | 3.3250 | 4.4600 | 67.8 | 149.2 |
| 81 | train-multi_query_aggregation-0263 | 5.8000 | 0.1732 | 56.2 | 126.2 |
| 82 | train-distractor_schema-0264 | 3.4250 | 3.5940 | 48.1 | 100.0 |
| 83 | train-multi_query_aggregation-0032 | -0.6250 | 3.7672 | 54.9 | 119.8 |
| 84 | train-multi_query_aggregation-0046 | -0.6250 | 2.5577 | 69.1 | 138.7 |
| 85 | train-tool_selection-0083 | 3.4250 | 3.5940 | 34.0 | 104.8 |
| 86 | train-error_recovery-0247 | 5.4500 | 0.0866 | 45.4 | 103.6 |
| 87 | train-sql_calculator-0024 | -0.7250 | 3.5940 | 61.6 | 110.3 |
| 88 | train-tool_selection-0125 | 3.4250 | 3.5940 | 35.8 | 105.5 |
| 89 | train-multi_query_aggregation-0228 | -2.0875 | 1.3221 | 59.5 | 139.6 |
| 90 | train-tool_selection-0013 | 3.4250 | 3.5940 | 33.5 | 106.9 |
| 91 | train-multi_query_aggregation-0025 | 0.0625 | 3.5738 | 70.7 | 148.6 |
| 92 | train-tool_selection-0027 | 3.4250 | 3.5940 | 29.5 | 105.4 |
| 93 | train-multi_query_aggregation-0298 | 6.0250 | 0.1250 | 73.8 | 145.8 |
| 94 | train-multi_table_reasoning-0057 | 5.5625 | 0.1083 | 55.6 | 98.9 |
| 95 | train-sql_calculator-0073 | -0.7250 | 3.5940 | 63.2 | 124.3 |
| 96 | train-sql_calculator-0171 | 1.3500 | 4.1500 | 64.0 | 133.4 |
| 97 | train-sql_calculator-0164 | 3.0250 | 4.2868 | 60.8 | 149.1 |
| 98 | train-tool_selection-0139 | 3.4250 | 3.5940 | 36.8 | 103.7 |
| 99 | train-multi_query_aggregation-0004 | 3.7250 | 3.7672 | 79.3 | 143.1 |
| 100 | train-multi_query_aggregation-0095 | -1.3000 | 1.5000 | 76.9 | 138.3 |

## Checkpoint List

- step 20: `/root/autodl-tmp/art-cp7/agentic-rl-grpo/models/cp7-formal-grpo-001/checkpoints/0020` (milestone)
- step 40: `/root/autodl-tmp/art-cp7/agentic-rl-grpo/models/cp7-formal-grpo-001/checkpoints/0040` (milestone)
- step 60: `/root/autodl-tmp/art-cp7/agentic-rl-grpo/models/cp7-formal-grpo-001/checkpoints/0060` (milestone)
- step 80: `/root/autodl-tmp/art-cp7/agentic-rl-grpo/models/cp7-formal-grpo-001/checkpoints/0080` (best-validation)
- step 100: `/root/autodl-tmp/art-cp7/agentic-rl-grpo/models/cp7-formal-grpo-001/checkpoints/0100` (last)

## Validation Metrics by Checkpoint

| Step | Success | Final acc. | Result | Grounded | Recovery | Avg steps | Avg tools | Max-step | Reward mean/std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 0.7267 | 0.7333 | 0.7733 | 0.7267 | 0.9697 | 5.07 | 4.12 | 0.0533 | 3.2720/3.7230 |
| 40 | 0.6933 | 0.7067 | 0.7467 | 0.6933 | 1.0000 | 4.50 | 3.52 | 0.0200 | 3.1363/3.7049 |
| 60 | 0.7467 | 0.7600 | 0.8000 | 0.7467 | 0.8333 | 4.25 | 3.25 | 0.0000 | 3.6120/3.3575 |
| 80 | 0.8867 | 0.8867 | 0.9133 | 0.8867 | 1.0000 | 4.07 | 3.07 | 0.0000 | 4.6630/2.4803 |
| 100 | 0.8733 | 0.8867 | 0.9333 | 0.8733 | 1.0000 | 4.00 | 3.00 | 0.0000 | 4.6903/2.2700 |

## Best Checkpoint Selection

Checkpoints are ranked by Task Success, then grounded rate, result correctness, final accuracy, lower invalid-call rate, and lower average steps.

Selected step: 80

## Failure Analysis

- arithmetic_error: 2
- excessive_tool_use: 9
- grounding_failure: 17
- hallucination: 17
- sql_error: 7

## Comparison against CP4 Baseline

- CP4 Task Success: 0.4933
- CP4 Final Accuracy: 0.5400
- CP4 Result Correctness: 0.5533
- CP4 Grounded Rate: 0.4933

Best validation Task Success delta: +0.3933

CP7 does not establish held-out improvement. CP8 frozen-test evaluation requires separate review and authorization.
