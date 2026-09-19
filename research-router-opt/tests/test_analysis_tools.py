from __future__ import annotations

import time

from research_router_opt.analysis_models import ToolCall
from research_router_opt.analysis_tools import AnalysisToolRegistry, create_analysis_database


def test_schema_sql_and_calculator_tools() -> None:
    registry = AnalysisToolRegistry(create_analysis_database("train_analytics"))
    try:
        schema = registry.execute(
            ToolCall("schema-1", "schema", {"table": "orders"}), timeout_s=1.0
        )
        assert schema.status == "ok"
        assert "orders" in schema.output["tables"]

        query = registry.execute(
            ToolCall("sql-1", "sql", {"query": "SELECT COUNT(*) AS value FROM orders"}),
            timeout_s=1.0,
        )
        assert query.status == "ok"
        assert query.output["rows"] == [[720]]

        calculation = registry.execute(
            ToolCall(
                "calc-1",
                "calculator",
                {
                    "expression": "(revenue - cost) / revenue",
                    "variables": {"revenue": 10, "cost": 4},
                },
            ),
            timeout_s=1.0,
        )
        assert calculation.status == "ok"
        assert calculation.output["value"] == 0.6
    finally:
        registry.close()


def test_sql_tool_rejects_write_and_reports_wrong_column() -> None:
    registry = AnalysisToolRegistry(create_analysis_database("train_analytics"))
    try:
        write = registry.execute(
            ToolCall("sql-write", "sql", {"query": "DELETE FROM orders"}), timeout_s=1.0
        )
        wrong_column = registry.execute(
            ToolCall("sql-bad", "sql", {"query": "SELECT area FROM customers"}),
            timeout_s=1.0,
        )
        wrong_table = registry.execute(
            ToolCall("sql-table", "sql", {"query": "SELECT * FROM missing_table"}),
            timeout_s=1.0,
        )
        syntax_error = registry.execute(
            ToolCall("sql-syntax", "sql", {"query": "SELECT FROM orders"}),
            timeout_s=1.0,
        )
        assert write.status == "error"
        assert "read-only" in (write.error or "")
        assert wrong_column.status == "error"
        assert wrong_table.status == "error"
        assert syntax_error.status == "error"
    finally:
        registry.close()


def test_malformed_call_unknown_tool_and_timeout() -> None:
    def slow_tool(arguments: dict[str, object]) -> dict[str, object]:
        del arguments
        time.sleep(0.05)
        return {"done": True}

    registry = AnalysisToolRegistry(
        create_analysis_database("train_analytics"), extra_tools={"slow": slow_tool}
    )
    try:
        malformed = registry.execute(
            ToolCall("sql-missing", "sql", {"unexpected": "value"}), timeout_s=1.0
        )
        unknown = registry.execute(ToolCall("unknown", "missing", {}), timeout_s=1.0)
        timed_out = registry.execute(ToolCall("slow", "slow", {}), timeout_s=0.001)
        assert malformed.status == "error"
        assert unknown.status == "invalid"
        assert timed_out.status == "timeout"
    finally:
        registry.close()
