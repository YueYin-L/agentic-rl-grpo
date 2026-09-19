# Held-out Failure Analysis

Failed tasks: 3 / 24

| Task | Query | Expected | Selected | Reward |
|---|---|---|---|---:|
| val-paper_search-00-00 | 不用总结，搜索发表过的亚洲象研究文献 | paper_search | abstract_summarize | -0.200 |
| val-paper_search-00-01 | 不用总结，搜索发表过的穿山甲研究文献 | paper_search | abstract_summarize | -0.200 |
| val-paper_search-00-02 | 不用总结，搜索发表过的长臂猿研究文献 | paper_search | abstract_summarize | -0.200 |

Observed pattern: all remaining errors contain the negated distractor `不用总结` before an explicit paper-search request. The bag-of-keywords state encoder detects `总结` but cannot represent negation or word order.

Next technical step: replace binary keyword counts with a learned text encoder or add explicit negation features, then evaluate on a newly frozen test set. The current validation set must not be reused for tuning.
