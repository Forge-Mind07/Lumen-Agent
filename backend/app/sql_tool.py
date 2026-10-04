from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import duckdb
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .datasets import json_safe


class UnsafeQueryError(ValueError):
    pass


DISALLOWED_EXPRESSIONS = (
    exp.Alter,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Merge,
    exp.Set,
    exp.Transaction,
    exp.Update,
    exp.Use,
)


@dataclass(slots=True)
class QueryResult:
    columns: list[str]
    column_types: list[str]
    rows: list[list[Any]]
    returned_row_count: int
    truncated: bool
    duration_ms: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "column_types": self.column_types,
            "rows": self.rows,
            "returned_row_count": self.returned_row_count,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
        }


def validate_read_only_sql(query: str) -> str:
    cleaned = query.strip().removesuffix(";").strip()
    if not cleaned:
        raise UnsafeQueryError("The SQL query is empty")
    try:
        statements = parse(cleaned, read="duckdb")
    except ParseError as exc:
        raise UnsafeQueryError(f"SQL could not be parsed: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError("Exactly one SQL statement is allowed")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise UnsafeQueryError("Only SELECT-style analytical queries are allowed")
    for node in statement.walk():
        if isinstance(node, DISALLOWED_EXPRESSIONS):
            raise UnsafeQueryError(f"{node.key.upper()} statements are not allowed")
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    for table in statement.find_all(exp.Table):
        table_name = table.name.lower()
        if table_name != "data" and table_name not in cte_names:
            raise UnsafeQueryError("Queries may only read the uploaded table named data")
    return cleaned


def execute_read_only_query(
    duckdb_path: str,
    query: str,
    *,
    row_limit: int,
    timeout_seconds: float,
) -> QueryResult:
    safe_query = validate_read_only_sql(query)
    connection = duckdb.connect(duckdb_path, read_only=True)
    timer = threading.Timer(timeout_seconds, connection.interrupt)
    started = time.perf_counter()
    try:
        timer.start()
        cursor = connection.execute(
            f"SELECT * FROM ({safe_query}) AS _agent_result LIMIT {row_limit + 1}"
        )
        columns = [column[0] for column in cursor.description]
        column_types = [str(column[1]) for column in cursor.description]
        raw_rows = cursor.fetchall()
    except duckdb.InterruptException as exc:
        raise TimeoutError(f"Query exceeded the {timeout_seconds:g}s timeout") from exc
    finally:
        timer.cancel()
        connection.close()
    duration_ms = round((time.perf_counter() - started) * 1000)
    truncated = len(raw_rows) > row_limit
    rows = [
        [json_safe(value) for value in row]
        for row in raw_rows[:row_limit]
    ]
    return QueryResult(
        columns=columns,
        column_types=column_types,
        rows=rows,
        returned_row_count=len(rows),
        truncated=truncated,
        duration_ms=duration_ms,
    )
