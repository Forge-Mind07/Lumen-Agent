from pathlib import Path

import duckdb
import pytest

from app.sql_tool import UnsafeQueryError, execute_read_only_query, validate_read_only_sql


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE data",
        "DELETE FROM data",
        "UPDATE data SET amount = 1",
        "INSERT INTO data VALUES (1)",
        "CREATE TABLE copied AS SELECT * FROM data",
        "SELECT 1; SELECT 2",
        "COPY data TO '/tmp/leak.csv'",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_rejects_non_read_only_sql(query: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_read_only_sql(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT count(*) FROM data",
        "WITH totals AS (SELECT sum(amount) AS value FROM data) SELECT * FROM totals",
        "SELECT region, sum(amount) FROM data GROUP BY region ORDER BY 2 DESC",
    ],
)
def test_accepts_analytical_sql(query: str) -> None:
    assert validate_read_only_sql(query)


def test_query_results_are_capped(tmp_path: Path) -> None:
    database = tmp_path / "test.duckdb"
    connection = duckdb.connect(database)
    connection.execute("CREATE TABLE data AS SELECT range AS value FROM range(100)")
    connection.close()

    result = execute_read_only_query(
        str(database), "SELECT * FROM data ORDER BY value", row_limit=10, timeout_seconds=2
    )

    assert result.returned_row_count == 10
    assert result.truncated is True
    assert result.rows[0] == [0]
    assert result.rows[-1] == [9]
