# Validation Comparison

| Metric | Keyword baseline | LinUCB router | Change |
|---|---:|---:|---:|
| Tool selection accuracy | 50.00% | 87.50% | +37.50% |
| Average reward | 0.400 | 0.850 | +0.450 |

The validation set uses templates and subjects excluded from training.
The policy updates only the lightweight routing layer; no LLM weights are trained.
