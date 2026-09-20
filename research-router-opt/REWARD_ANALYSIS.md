# CP5 Offline Reward Analysis

## Signal Statistics

| Signal | Activation | P(signal\|success) | P(signal\|failure) | Variance |
|---|---:|---:|---:|---:|
| answer_correct | 0.5400 | 1.0000 | 0.0921 | 0.2484 |
| result_correct | 0.5533 | 1.0000 | 0.1184 | 0.2472 |
| grounded | 0.4933 | 1.0000 | 0.0000 | 0.2500 |
| tool_valid | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| sql_exec_success | 0.9200 | 1.0000 | 0.8421 | 0.0736 |
| recovered_from_error | 0.2133 | 0.0946 | 0.3289 | 0.1678 |
| meaningful_recovery | 0.0467 | 0.0946 | 0.0000 | 0.0445 |
| redundant_calls | 0.0467 | 0.0135 | 0.0789 | 0.0445 |
| max_step_termination | 0.1133 | 0.0000 | 0.2237 | 0.1005 |
| tool_error | 0.2333 | 0.0946 | 0.3684 | 0.1789 |
| sql_error | 0.0333 | 0.0270 | 0.0395 | 0.0322 |
| calculator_error | 0.0600 | 0.0135 | 0.1053 | 0.0564 |
| multi_observation_completion | 0.0667 | 0.0270 | 0.1053 | 0.0622 |

`tool_valid` is saturated and `sql_exec_success` is near-saturated; neither receives positive canonical reward. Full signal overlap and per-task-type activation are saved in `signal_statistics.json`.

## Recovery Audit

- Error opportunities: 35
- Broad recovered_from_error: 32
- Meaningful recovery: 7
- Superficial recovery: 25
- Meaningful recovery requires a changed action for the failed tool plus correct, grounded downstream evidence. An unrelated successful call is not rewarded.

## Reward Formula

```text
R = 2.0*answer_correct + 2.0*result_correct + 1.5*grounded
  + 0.4*meaningful_recovery + 0.4*valid_multi_observation
  - 1.0*wrong_answer - 1.0*wrong_result - 0.8*ungrounded
  - bounded(tool_errors, redundant_calls, repeated missing-object probes,
            excess calls, max-step termination)
```

Tool validity and SQL execution are guardrails, not positive farming targets. Necessary Schema -> SQL -> Calculator steps are not linearly penalized.

## Reward Component Explanation

- Outcome correctness contributes at most 5.5 positive points and dominates process credit (at most 0.8).
- Meaningful recovery is rewarded only after relevant correction and grounded success.
- Multi-observation credit requires two successful SQL observations, Calculator, correct result, and grounding.
- Penalties are capped so one malformed episode cannot create unbounded reward scale.

## Reward Distribution

- Overall: mean=1.2500, std=4.3504, p25=-2.9500, median=0.2000, p75=5.5000.
- Success mean: 5.4750.
- Failure mean: -2.8638.

## Success vs Failure Separation

- success_gt_failure: PASS; 5.4750 > -2.8638.
- efficient_gt_inefficient_success: PASS; 5.5232 > 5.3250.
- grounded_gt_unsupported_correct: PASS; 5.4750 > 0.0357.
- correct_gt_wrong_arithmetic: PASS; 5.1000 > -3.0265.
- meaningful_gt_superficial_recovery: PASS; 5.5286 > -3.9100.

## Task-type Analysis

- distractor_schema: n=21, mean=1.6690, std=3.9405.
- error_recovery: n=22, mean=4.3091, std=3.0331.
- multi_query_aggregation: n=21, mean=-2.2429, std=3.6526.
- multi_table_reasoning: n=22, mean=1.1114, std=3.7351.
- schema_discovery: n=22, mean=5.5000, std=0.0000.
- sql_calculator: n=21, mean=-2.9952, std=2.0430.
- tool_selection: n=21, mean=1.0571, std=4.3250.

## Pairwise Sanity Cases

- successful: n=74, mean=5.4750, examples=validation-schema_discovery-0000, validation-multi_table_reasoning-0001, validation-distractor_schema-0005.
- failed: n=76, mean=-2.8638, examples=validation-error_recovery-0002, validation-sql_calculator-0003, validation-multi_query_aggregation-0004.
- efficient_success: n=56, mean=5.5232, examples=validation-schema_discovery-0000, validation-multi_table_reasoning-0001, validation-schema_discovery-0007.
- inefficient_success: n=18, mean=5.3250, examples=validation-distractor_schema-0005, validation-tool_selection-0006, validation-tool_selection-0013.
- grounded_correct: n=74, mean=5.4750, examples=validation-schema_discovery-0000, validation-multi_table_reasoning-0001, validation-distractor_schema-0005.
- unsupported_correct: n=7, mean=0.0357, examples=validation-multi_table_reasoning-0022, validation-multi_table_reasoning-0050, validation-multi_table_reasoning-0085.
- correct_arithmetic: n=1, mean=5.1000, examples=validation-sql_calculator-0031.
- wrong_arithmetic: n=17, mean=-3.0265, examples=validation-sql_calculator-0010, validation-sql_calculator-0017, validation-sql_calculator-0024.
- meaningful_recovery: n=7, mean=5.5286, examples=validation-tool_selection-0006, validation-tool_selection-0013, validation-sql_calculator-0031.
- superficial_recovery: n=25, mean=-3.9100, examples=validation-error_recovery-0002, validation-multi_query_aggregation-0004, validation-sql_calculator-0010.

## Candidate Comparison

| Candidate | Success mean | Failure mean | Separation | Max failed reward |
|---|---:|---:|---:|---:|
| outcome_guardrail | 5.4264 | -2.8638 | 8.2902 | 0.2000 |
| balanced | 5.4750 | -2.8638 | 8.3388 | 0.2000 |
| process_richer | 5.5291 | -2.8224 | 8.3514 | 0.2000 |

`balanced` is frozen as canonical: it preserves outcome dominance, passes all pairwise orderings, and gives limited process credit without making error recovery or long trajectories profitable.

## Reward Hacking Audit

- Highest failed reward: 0.2000.
- Lowest successful reward: 5.0500.
- Tool/schema/calculator spam cannot add positive reward; it adds bounded penalties.
- SQL execution and tool validity provide no positive reward.
- Wrong arithmetic receives no answer/grounding credit and remains far below correct arithmetic despite executable tools.
- Main residual risk: strict answer formatting and verifier coverage can still shape outcome reward; monitor validation success beside training reward in CP6.

Top high-reward failures:

- validation-multi_table_reasoning-0022: reward=0.2000, type=multi_table_reasoning, signals=['answer_correct', 'tool_valid', 'sql_exec_success'].
- validation-multi_query_aggregation-0025: reward=0.2000, type=multi_query_aggregation, signals=['result_correct', 'tool_valid', 'sql_exec_success', 'multi_observation_completion'].
- validation-multi_query_aggregation-0046: reward=0.2000, type=multi_query_aggregation, signals=['result_correct', 'tool_valid', 'sql_exec_success', 'multi_observation_completion'].
- validation-multi_table_reasoning-0050: reward=0.2000, type=multi_table_reasoning, signals=['answer_correct', 'tool_valid', 'sql_exec_success'].
- validation-multi_table_reasoning-0085: reward=0.2000, type=multi_table_reasoning, signals=['answer_correct', 'tool_valid', 'sql_exec_success'].

## Representative Trajectories

- Meaningful recovery and superficial recovery examples are recorded in `audit.json` and the pairwise section above.
- Per-trajectory explanations are saved in `reward_decomposition.jsonl`; each row contains outcome, process, penalties, counts, and total reward.

## CP5 Status

**CP5 PASS**

Canonical reward is frozen for CP6 Minimal GRPO Smoke Test only. No formal GRPO training or frozen-test evaluation has started.
