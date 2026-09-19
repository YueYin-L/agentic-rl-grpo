"""Deterministic DuckDB-backed tools for multi-turn data-analysis tasks."""

from __future__ import annotations

import ast
import operator
import random
import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import date, timedelta
from decimal import Decimal
from time import perf_counter
from typing import Any

import duckdb

from research_router_opt.analysis_models import ToolCall, ToolResult

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

_WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|copy|export|import|"
    r"install|load|call|pragma|vacuum)\b",
    flags=re.IGNORECASE,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, Decimal)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _fixture_seed(database_id: str) -> int:
    seeds = {
        "train_analytics": 20260919,
        "validation_analytics": 20260920,
        "test_analytics": 20260921,
    }
    try:
        return seeds[database_id]
    except KeyError as exc:
        raise ValueError(f"Unknown database fixture: {database_id}") from exc


def create_analysis_database(database_id: str) -> duckdb.DuckDBPyConnection:
    """Create a split-specific in-memory database with stable but isolated rows."""

    randomizer = random.Random(_fixture_seed(database_id))
    regions = ("north", "south", "east", "west", "central")
    segments = ("consumer", "enterprise", "education")
    categories = ("sensor", "camera", "compute", "storage")
    statuses = ("completed", "completed", "completed", "refunded", "cancelled")

    customers = [
        {
            "customer_id": customer_id,
            "region": regions[(customer_id + randomizer.randrange(len(regions))) % len(regions)],
            "segment": segments[
                (customer_id + randomizer.randrange(len(segments))) % len(segments)
            ],
            "signup_year": 2021 + customer_id % 5,
        }
        for customer_id in range(1, 91)
    ]
    products = [
        {
            "product_id": product_id,
            "category": categories[(product_id - 1) % len(categories)],
            "unit_cost": float(20 + product_id * 7),
        }
        for product_id in range(1, 17)
    ]
    start = date(2024, 1, 1)
    orders: list[dict[str, Any]] = []
    refunds: list[dict[str, Any]] = []
    for order_id in range(1, 721):
        product_id = 1 + randomizer.randrange(len(products))
        quantity = 1 + randomizer.randrange(5)
        unit_price = float(55 + product_id * 13 + randomizer.randrange(25))
        status = statuses[randomizer.randrange(len(statuses))]
        order_date = start + timedelta(days=randomizer.randrange(730))
        orders.append(
            {
                "order_id": order_id,
                "customer_id": 1 + randomizer.randrange(len(customers)),
                "product_id": product_id,
                "order_date": order_date,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": round(randomizer.choice((0.0, 0.05, 0.1, 0.15)), 2),
                "status": status,
            }
        )
        if status == "refunded":
            refunds.append(
                {
                    "refund_id": len(refunds) + 1,
                    "order_id": order_id,
                    "refund_amount": round(unit_price * quantity * randomizer.uniform(0.4, 1.0), 2),
                }
            )

    campaigns = [
        {
            "campaign_id": index,
            "area": regions[index % len(regions)],
            "spend": float(1000 + index * 125),
        }
        for index in range(1, 21)
    ]

    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE customers(customer_id INTEGER, region VARCHAR, segment VARCHAR, "
        "signup_year INTEGER)"
    )
    connection.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [
            (row["customer_id"], row["region"], row["segment"], row["signup_year"])
            for row in customers
        ],
    )
    connection.execute(
        "CREATE TABLE products(product_id INTEGER, category VARCHAR, unit_cost DOUBLE)"
    )
    connection.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [(row["product_id"], row["category"], row["unit_cost"]) for row in products],
    )
    connection.execute(
        "CREATE TABLE orders(order_id INTEGER, customer_id INTEGER, product_id INTEGER, "
        "order_date DATE, quantity INTEGER, unit_price DOUBLE, discount DOUBLE, status VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["order_id"],
                row["customer_id"],
                row["product_id"],
                row["order_date"],
                row["quantity"],
                row["unit_price"],
                row["discount"],
                row["status"],
            )
            for row in orders
        ],
    )
    connection.execute(
        "CREATE TABLE refunds(refund_id INTEGER, order_id INTEGER, refund_amount DOUBLE)"
    )
    connection.executemany(
        "INSERT INTO refunds VALUES (?, ?, ?)",
        [(row["refund_id"], row["order_id"], row["refund_amount"]) for row in refunds],
    )
    connection.execute(
        "CREATE TABLE marketing_campaigns(campaign_id INTEGER, area VARCHAR, spend DOUBLE)"
    )
    connection.executemany(
        "INSERT INTO marketing_campaigns VALUES (?, ?, ?)",
        [(row["campaign_id"], row["area"], row["spend"]) for row in campaigns],
    )
    return connection


def _evaluate_expression(expression: str, variables: Mapping[str, float]) -> float:
    binary_ops: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown calculator variable: {node.id}")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            return binary_ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](visit(node.operand))
        raise ValueError("Calculator expression contains an unsupported operation.")

    if len(expression) > 300:
        raise ValueError("Calculator expression is too long.")
    parsed = ast.parse(expression, mode="eval")
    result = visit(parsed)
    if abs(result) > 1e18:
        raise ValueError("Calculator result exceeds the allowed range.")
    return result


class AnalysisToolRegistry:
    """Schema, read-only SQL, and safe arithmetic tools over one episode database."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        extra_tools: Mapping[str, ToolHandler] | None = None,
    ) -> None:
        self.connection = connection
        self._handlers: dict[str, ToolHandler] = {
            "schema": self._schema,
            "sql": self._sql,
            "calculator": self._calculator,
        }
        if extra_tools:
            self._handlers.update(extra_tools)

    @property
    def tool_specs(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "schema",
                    "description": "Inspect available tables and columns before querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {"table": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sql",
                    "description": "Execute one read-only DuckDB SELECT or WITH query.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate safe arithmetic using named numeric variables.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "variables": {"type": "object"},
                        },
                        "required": ["expression"],
                    },
                },
            },
        )

    def close(self) -> None:
        self.connection.close()

    def execute(self, call: ToolCall, *, timeout_s: float) -> ToolResult:
        started = perf_counter()
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="invalid",
                error=f"Unknown tool: {call.name}",
            )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-tool")
        future = executor.submit(handler, call.arguments)
        try:
            output = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            future.cancel()
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="timeout",
                error=f"Tool exceeded timeout of {timeout_s:.3f}s",
                latency_ms=(perf_counter() - started) * 1000,
            )
        except (ValueError, duckdb.Error, SyntaxError, ZeroDivisionError) as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=str(exc),
                latency_ms=(perf_counter() - started) * 1000,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="ok",
            output=output,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def _schema(self, arguments: dict[str, Any]) -> dict[str, Any]:
        table = arguments.get("table")
        if table is not None and not isinstance(table, str):
            raise ValueError("schema.table must be a string.")
        query = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
        """
        parameters: list[Any] = []
        if table:
            query += " AND table_name = ?"
            parameters.append(table)
        query += " ORDER BY table_name, ordinal_position"
        rows = self.connection.execute(query, parameters).fetchall()
        if table and not rows:
            raise ValueError(f"Unknown table: {table}")
        tables: dict[str, list[dict[str, str]]] = {}
        for table_name, column_name, data_type in rows:
            tables.setdefault(str(table_name), []).append(
                {"name": str(column_name), "type": str(data_type)}
            )
        return {"tables": tables}

    def _sql(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("sql.query must be a non-empty string.")
        normalized = query.strip().rstrip(";").strip()
        if ";" in normalized:
            raise ValueError("Only one SQL statement is allowed.")
        if _WRITE_SQL.search(normalized) or not re.match(r"^(select|with)\b", normalized, re.I):
            raise ValueError("Only read-only SELECT or WITH queries are allowed.")
        cursor = self.connection.execute(normalized)
        columns = [str(description[0]) for description in cursor.description]
        rows = [[_json_value(value) for value in row] for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}

    def _calculator(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = arguments.get("expression")
        variables = arguments.get("variables", {})
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("calculator.expression must be a non-empty string.")
        if not isinstance(variables, dict) or not all(
            isinstance(key, str) and isinstance(value, (int, float))
            for key, value in variables.items()
        ):
            raise ValueError("calculator.variables must map names to numbers.")
        numeric_variables = {key: float(value) for key, value in variables.items()}
        return {"value": _evaluate_expression(expression, numeric_variables)}
