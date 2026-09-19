"""Deterministic local tools keep the RL environment cheap and reproducible."""

from __future__ import annotations

from collections.abc import Callable

from research_router_opt.models import ToolObservation


def species_lookup(query: str) -> ToolObservation:
    return ToolObservation(
        tool="species_lookup",
        status="ok",
        result={"kind": "species_card", "query": query, "source": "local_fixture"},
    )


def paper_search(query: str) -> ToolObservation:
    return ToolObservation(
        tool="paper_search",
        status="ok",
        result={"kind": "paper_candidates", "query": query, "count": 3},
    )


def evidence_compare(query: str) -> ToolObservation:
    return ToolObservation(
        tool="evidence_compare",
        status="ok",
        result={"kind": "evidence_matrix", "query": query, "dimensions": 3},
    )


def abstract_summarize(query: str) -> ToolObservation:
    return ToolObservation(
        tool="abstract_summarize",
        status="ok",
        result={"kind": "structured_summary", "query": query, "sections": 3},
    )


TOOLS: dict[str, Callable[[str], ToolObservation]] = {
    "species_lookup": species_lookup,
    "paper_search": paper_search,
    "evidence_compare": evidence_compare,
    "abstract_summarize": abstract_summarize,
}


def execute_tool(tool_name: str, query: str) -> ToolObservation:
    try:
        tool = TOOLS[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {tool_name}") from exc
    return tool(query)

