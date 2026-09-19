# Multi-turn Data-analysis Task Specification

## Objective

Measure whether a model can solve grounded data-analysis questions through a sequence of schema,
SQL, and calculator actions, including observing errors and replanning. Tasks are not scored by SQL
string matching; they are scored by execution results and complete trajectory behavior.

## Tool Contract

| Tool | Input | Output | Boundary |
|---|---|---|---|
| `schema` | optional table name | tables, columns, types | read-only metadata |
| `sql` | one query | columns, rows, row count | one `SELECT`/`WITH`; writes rejected |
| `calculator` | expression and numeric variables | numeric value | arithmetic AST only |

Every tool result has `ok`, `error`, `invalid`, or `timeout` status and is returned to the model as
an observation. Errors are part of the trajectory rather than Python exceptions escaping the runtime.

## Taxonomy

| Type | Required behavior | Typical minimum path |
|---|---|---|
| Schema Discovery | discover real columns before query | Schema → SQL → Final |
| Multi-table Reasoning | join customers, products, and orders | Schema → SQL → Final |
| Error Recovery | correct an obsolete/wrong field after feedback | SQL error → Schema → SQL → Final |
| SQL + Calculator | separate retrieval from arithmetic | Schema → SQL → Calculator → Final |
| Multi-query Aggregation | combine independent query observations | Schema → SQL → SQL → Calculator → Final |
| Distractor Schema | avoid a plausible but irrelevant table | Schema → SQL → Final |
| Tool Selection | choose SQL for retrieval and calculator for arithmetic | Schema → SQL → Calculator → Final |

The expected sequence describes task intent, not an exact action-string requirement. Equivalent
read-only SQL and efficient valid paths remain acceptable.

## Data Fixtures

Each split has a distinct deterministic database seed and database ID. Fixtures contain:

- `customers(customer_id, region, segment, signup_year)`
- `products(product_id, category, unit_cost)`
- `orders(order_id, customer_id, product_id, order_date, quantity, unit_price, discount, status)`
- `refunds(refund_id, order_id, refund_amount)`
- `marketing_campaigns(campaign_id, area, spend)` as a distractor

## Splits

| Split | Tasks | Database ID | Use |
|---|---:|---|---|
| Train | 300 | `train_analytics` | rollout and policy update |
| Validation | 150 | `validation_analytics` | baseline, reward audit, model selection |
| Test | 150 | `test_analytics` | one final frozen evaluation |

All seven task types appear in every split. IDs, database rows, and wording prefixes are split-specific.
The exact serialized hashes are stored in `data/analysis_manifest.json`.

## Ground Truth

Each task stores:

- expected final answer and numeric tolerance;
- one or more executable reference SQL queries;
- semantic result rows for each reference query;
- expected tool capabilities;
- whether error recovery is central to the scenario.

Reference SQL is used to construct ground truth, not for exact-string scoring. A candidate query is
correct when its rows are semantically equal to the required result within tolerance.

## Programmatic Verification

The verifier emits:

- `answer_correct`
- `sql_exec_success`
- `result_correct`
- `tool_valid`
- `schema_valid`
- `grounded`
- `recovered_from_error`
- `step_count`
- `redundant_calls`
- explicit failure modes

`grounded` requires the correct final answer to be present in a successful tool observation. A fluent
answer without supporting tool output therefore fails.

## Leakage Prevention

- Test data, wording, and fixture seed are distinct and hash-recorded.
- Test must not be used for prompt, reward, difficulty, hyperparameter, or stopping decisions.
- Validation diagnostics may change the task generator only before the formal baseline; any such
  change requires regenerating all hashes and documenting the new dataset version.
- Expected outputs must never be edited to make a failing trajectory pass.

## Current Limitations

- Data is synthetic and intentionally bounded; external benchmark expansion is deferred.
- Template families share analytical structure across splits, so CP4 must check for wording shortcuts.
- Tool-use necessity has been designed and CPU-tested, but has not yet been demonstrated with the
  target real model.
