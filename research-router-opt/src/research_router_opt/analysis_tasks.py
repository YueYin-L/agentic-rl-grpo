"""Deterministic multi-turn task generation with frozen split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import duckdb

from research_router_opt.analysis_models import AnalysisTask, SplitName
from research_router_opt.analysis_tools import create_analysis_database

TASK_TAXONOMY = (
    "schema_discovery",
    "multi_table_reasoning",
    "error_recovery",
    "sql_calculator",
    "multi_query_aggregation",
    "distractor_schema",
    "tool_selection",
)

DEFAULT_SPLIT_SIZES: dict[SplitName, int] = {
    "train": 300,
    "validation": 150,
    "test": 150,
}

_DATABASE_IDS: dict[SplitName, str] = {
    "train": "train_analytics",
    "validation": "validation_analytics",
    "test": "test_analytics",
}

_SPLIT_WORDING: dict[SplitName, tuple[str, ...]] = {
    "train": ("请分析", "计算", "查询并回答"),
    "validation": ("请核验", "基于数据库判断", "通过工具分析"),
    "test": ("独立分析", "请给出数据依据", "检查数据后回答"),
}


def _rows(connection: duckdb.DuckDBPyConnection, query: str) -> tuple[tuple[Any, ...], ...]:
    raw_rows = connection.execute(query).fetchall()
    return tuple(
        tuple(value.item() if hasattr(value, "item") else value for value in row)
        for row in raw_rows
    )


def _number(value: Any) -> float:
    return float(cast(int | float, value))


def build_analysis_tasks(split: SplitName, count: int | None = None) -> list[AnalysisTask]:
    target_count = DEFAULT_SPLIT_SIZES[split] if count is None else count
    if target_count < len(TASK_TAXONOMY):
        raise ValueError(f"count must be at least {len(TASK_TAXONOMY)} to cover the taxonomy.")
    database_id = _DATABASE_IDS[split]
    connection = create_analysis_database(database_id)
    regions = ("north", "south", "east", "west", "central")
    segments = ("consumer", "enterprise", "education")
    categories = ("sensor", "camera", "compute", "storage")
    statuses = ("completed", "refunded", "cancelled")
    wording = _SPLIT_WORDING[split]
    tasks: list[AnalysisTask] = []
    try:
        for index in range(target_count):
            task_type = TASK_TAXONOMY[index % len(TASK_TAXONOMY)]
            variant = index // len(TASK_TAXONOMY)
            region = regions[variant % len(regions)]
            other_region = regions[(regions.index(region) + 2) % len(regions)]
            segment = segments[(variant // 3) % len(segments)]
            category = categories[(variant // 5) % len(categories)]
            status = statuses[(variant // 4) % len(statuses)]
            year = 2024 + ((variant // 10) % 2)
            quarter = 1 + variant % 4
            minimum_order_id = 1 + (variant * 13) % 200
            prefix = wording[index % len(wording)]
            task = _build_one_task(
                connection=connection,
                split=split,
                database_id=database_id,
                index=index,
                task_type=task_type,
                prefix=prefix,
                region=region,
                other_region=other_region,
                segment=segment,
                category=category,
                status=status,
                year=year,
                quarter=quarter,
                minimum_order_id=minimum_order_id,
            )
            tasks.append(task)
    finally:
        connection.close()
    return tasks


def _build_one_task(
    *,
    connection: duckdb.DuckDBPyConnection,
    split: SplitName,
    database_id: str,
    index: int,
    task_type: str,
    prefix: str,
    region: str,
    other_region: str,
    segment: str,
    category: str,
    status: str,
    year: int,
    quarter: int,
    minimum_order_id: int,
) -> AnalysisTask:
    task_id = f"{split}-{task_type}-{index:04d}"
    results: tuple[tuple[tuple[Any, ...], ...], ...]
    if task_type == "schema_discovery":
        sql = f"""
            SELECT COUNT(*) AS value FROM orders
            WHERE status = '{status}' AND YEAR(order_date) = {year}
              AND QUARTER(order_date) = {quarter} AND order_id >= {minimum_order_id}
        """
        results = (_rows(connection, sql),)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年第 {quarter} 季度 orders 表中状态为 {status} 的订单数；"
                f"仅统计订单号不小于 {minimum_order_id} 的记录，并先确认表结构。"
            ),
            expected_answer=int(results[0][0][0]),
            reference_sql=(sql,),
            expected_sql_results=results,
            expected_tools=("schema", "sql"),
        )

    if task_type == "multi_table_reasoning":
        sql = f"""
            SELECT ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS value
            FROM orders o
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE c.region = '{region}' AND p.category = '{category}'
              AND o.status = 'completed' AND YEAR(o.order_date) = {year}
              AND o.order_id >= {minimum_order_id}
        """
        results = (_rows(connection, sql),)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年 {region} 区域 {category} 类别已完成订单的折后收入，"
                f"仅统计订单号不小于 {minimum_order_id} 的记录，需要关联客户、产品与订单表。"
            ),
            expected_answer=round(_number(results[0][0][0]), 2),
            reference_sql=(sql,),
            expected_sql_results=results,
            expected_tools=("schema", "sql"),
        )

    if task_type == "error_recovery":
        sql = f"""
            SELECT COUNT(*) AS value
            FROM orders o JOIN customers c USING (customer_id)
            WHERE c.region = '{region}' AND YEAR(o.order_date) = {year}
              AND QUARTER(o.order_date) = {quarter}
              AND o.order_id >= {minimum_order_id}
        """
        results = (_rows(connection, sql),)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年第 {quarter} 季度 {region} 客户的订单数。"
                "旧报表把客户区域称为 area，"
                f"仅统计订单号不小于 {minimum_order_id} 的记录；请以实际 schema 为准，"
                "并能处理字段错误。"
            ),
            expected_answer=int(results[0][0][0]),
            reference_sql=(sql,),
            expected_sql_results=results,
            expected_tools=("schema", "sql"),
            requires_error_recovery=True,
        )

    if task_type == "sql_calculator":
        sql = f"""
            SELECT
              ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 4) AS revenue,
              ROUND(SUM(o.quantity * p.unit_cost), 4) AS cost
            FROM orders o
            JOIN products p USING (product_id)
            JOIN customers c USING (customer_id)
            WHERE p.category = '{category}' AND c.region = '{region}'
              AND o.status = 'completed' AND YEAR(o.order_date) = {year}
              AND o.order_id >= {minimum_order_id}
        """
        results = (_rows(connection, sql),)
        revenue = _number(results[0][0][0])
        cost = _number(results[0][0][1])
        margin = round((revenue - cost) / revenue * 100, 6)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年 {region} 区域 {category} 类别已完成订单的利润率百分比："
                f"仅统计订单号不小于 {minimum_order_id} 的记录；先查询折后收入和成本，"
                "再用计算器计算。"
            ),
            expected_answer=margin,
            reference_sql=(sql,),
            expected_sql_results=results,
            expected_tools=("schema", "sql", "calculator"),
            answer_tolerance=1e-4,
        )

    if task_type == "multi_query_aggregation":
        first_sql = f"""
            SELECT SUM(CASE WHEN o.status = 'refunded' THEN 1 ELSE 0 END)::DOUBLE
                   / COUNT(*) AS value
            FROM orders o
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE c.region = '{region}' AND p.category = '{category}'
              AND YEAR(o.order_date) = {year}
              AND o.order_id >= {minimum_order_id}
        """
        second_sql = f"""
            SELECT SUM(CASE WHEN o.status = 'refunded' THEN 1 ELSE 0 END)::DOUBLE
                   / COUNT(*) AS value
            FROM orders o
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE c.region = '{other_region}' AND p.category = '{category}'
              AND YEAR(o.order_date) = {year}
              AND o.order_id >= {minimum_order_id}
        """
        results = (_rows(connection, first_sql), _rows(connection, second_sql))
        difference = round((_number(results[0][0][0]) - _number(results[1][0][0])) * 100, 6)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年 {category} 类别中 {region} 与 {other_region} 的"
                f"退款订单率差值（百分点）；仅统计订单号不小于 {minimum_order_id} 的记录。"
                "分别查询两个区域，再用计算器求差。"
            ),
            expected_answer=difference,
            reference_sql=(first_sql, second_sql),
            expected_sql_results=results,
            expected_tools=("schema", "sql", "sql", "calculator"),
            answer_tolerance=1e-4,
        )

    if task_type == "distractor_schema":
        sql = f"""
            SELECT COUNT(*) AS value
            FROM orders o
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE c.region = '{region}' AND p.category = '{category}'
              AND YEAR(o.order_date) = {year}
              AND o.order_id >= {minimum_order_id}
        """
        results = (_rows(connection, sql),)
        return AnalysisTask(
            task_id=task_id,
            split=split,
            task_type=task_type,
            database_id=database_id,
            question=(
                f"{prefix} {year} 年 {region} 客户购买 {category} 产品的订单数。"
                f"仅统计订单号不小于 {minimum_order_id} 的记录；不要误用同样包含区域信息的 "
                "marketing_campaigns 表。"
            ),
            expected_answer=int(results[0][0][0]),
            reference_sql=(sql,),
            expected_sql_results=results,
            expected_tools=("schema", "sql"),
        )

    sql = f"""
        SELECT ROUND(AVG(quantity * unit_price * (1 - discount)), 6) AS value
        FROM orders o JOIN customers c USING (customer_id)
        WHERE o.status = '{status}' AND c.segment = '{segment}'
          AND YEAR(o.order_date) = {year}
          AND o.order_id >= {minimum_order_id}
    """
    results = (_rows(connection, sql),)
    adjusted = round(_number(results[0][0][0]) * 1.06, 6)
    return AnalysisTask(
        task_id=task_id,
        split=split,
        task_type=task_type,
        database_id=database_id,
        question=(
            f"{prefix} {year} 年 {segment} 客户的 {status} 订单平均折后金额，"
            f"仅统计订单号不小于 {minimum_order_id} 的记录，并用计算器给出上浮 6% 后的结果。"
        ),
        expected_answer=adjusted,
        reference_sql=(sql,),
        expected_sql_results=results,
        expected_tools=("schema", "sql", "calculator"),
        answer_tolerance=1e-4,
    )


def write_analysis_datasets(output_dir: Path) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for split, count in DEFAULT_SPLIT_SIZES.items():
        tasks = build_analysis_tasks(split, count)
        path = output_dir / f"analysis_{split}.jsonl"
        content = "".join(
            json.dumps(task.to_dict(), ensure_ascii=False, default=str) + "\n" for task in tasks
        )
        path.write_bytes(content.encode("utf-8"))
        manifest[split] = {
            "path": path.name,
            "task_count": len(tasks),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "database_id": _DATABASE_IDS[split],
        }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_analysis_tasks(path: Path) -> list[AnalysisTask]:
    tasks: list[AnalysisTask] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        tasks.append(
            AnalysisTask(
                task_id=payload["task_id"],
                split=payload["split"],
                task_type=payload["task_type"],
                database_id=payload["database_id"],
                question=payload["question"],
                expected_answer=payload["expected_answer"],
                reference_sql=tuple(payload["reference_sql"]),
                expected_sql_results=tuple(
                    tuple(tuple(row) for row in result)
                    for result in payload["expected_sql_results"]
                ),
                expected_tools=tuple(payload["expected_tools"]),
                requires_error_recovery=payload["requires_error_recovery"],
                answer_tolerance=float(payload["answer_tolerance"]),
            )
        )
    return tasks
