"""Deterministic query features used by the contextual-bandit policy."""

from __future__ import annotations

FEATURE_NAMES = (
    "bias",
    "species_terms",
    "paper_terms",
    "compare_terms",
    "summary_terms",
    "multiple_terms",
    "evidence_terms",
    "query_length",
)

_GROUPS: tuple[tuple[str, ...], ...] = (
    ("物种", "分布", "保护级别", "学名", "栖息地", "雪豹", "熊猫", "金丝猴"),
    ("论文", "文献", "研究进展", "检索", "搜索", "发表"),
    ("比较", "对比", "差异", "哪个更", "优缺点"),
    ("总结", "概括", "摘要", "提炼", "归纳"),
    ("两篇", "多篇", "多个", "分别", "之间"),
    ("证据", "结论", "方法", "实验", "引用"),
)


def extract_features(query: str) -> list[float]:
    """Map a Chinese research query to a compact, normalized feature vector."""

    normalized = query.strip().lower()
    if not normalized:
        raise ValueError("Query must not be empty.")

    values = [1.0]
    for terms in _GROUPS:
        values.append(float(sum(term in normalized for term in terms)))
    values.append(min(len(normalized) / 50.0, 1.0))
    return values

