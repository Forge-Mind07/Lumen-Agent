from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, BinaryIO

import duckdb

from .config import Settings
from .models import ColumnProfile, DatasetProfile, DatasetRecord, new_id
from .storage import Storage

ALLOWED_EXTENSIONS = {".csv", ".parquet"}
NUMERIC_TYPES = re.compile(
    r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|"
    r"FLOAT|DOUBLE|DECIMAL|REAL)"
)


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_upload(source: BinaryIO, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"File exceeds the {max_bytes:,}-byte upload limit")
            output.write(chunk)
    return size


def register_dataset(
    *,
    filename: str,
    source: BinaryIO,
    storage: Storage,
    settings: Settings,
) -> DatasetRecord:
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only CSV and Parquet files are supported in this version")

    dataset_id = new_id("ds")
    dataset_dir = settings.data_dir / "datasets" / dataset_id
    original_path = (dataset_dir / f"source{extension}").resolve()
    duckdb_path = (dataset_dir / "dataset.duckdb").resolve()
    size = save_upload(source, original_path, settings.max_upload_bytes)
    record = storage.create_dataset(
        dataset_id=dataset_id,
        filename=safe_name,
        file_type=extension.removeprefix("."),
        size_bytes=size,
        original_path=str(original_path),
        duckdb_path=str(duckdb_path),
    )

    try:
        profile = create_duckdb_and_profile(record, settings.max_profile_columns)
        return storage.finish_dataset(dataset_id, profile)
    except Exception as exc:
        storage.fail_dataset(dataset_id, friendly_duckdb_error(exc))
        raise ValueError(friendly_duckdb_error(exc)) from exc


def create_duckdb_and_profile(record: DatasetRecord, max_profile_columns: int) -> DatasetProfile:
    source = quote_literal(record.original_path)
    reader = (
        f"read_parquet({source})"
        if record.file_type == "parquet"
        else f"read_csv_auto({source}, sample_size=100000, header=true)"
    )
    connection = duckdb.connect(record.duckdb_path)
    try:
        connection.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM {reader}")
        return profile_table(connection, max_profile_columns)
    finally:
        connection.close()


def profile_table(connection: duckdb.DuckDBPyConnection, max_columns: int) -> DatasetProfile:
    schema_rows = connection.execute("DESCRIBE data").fetchall()
    row_count = connection.execute("SELECT count(*) FROM data").fetchone()[0]
    selected = schema_rows[:max_columns]
    expressions: list[str] = []
    for name, data_type, *_ in selected:
        column = quote_identifier(name)
        expressions.extend(
            [
                f"round(100.0 * count(*) FILTER (WHERE {column} IS NULL) / "
                f"greatest(count(*), 1), 2) AS {quote_identifier(name + '__null')}",
                f"approx_count_distinct({column}) AS {quote_identifier(name + '__distinct')}",
            ]
        )
        if NUMERIC_TYPES.match(data_type) or data_type.startswith(("DATE", "TIMESTAMP")):
            expressions.extend(
                [
                    f"min({column}) AS {quote_identifier(name + '__min')}",
                    f"max({column}) AS {quote_identifier(name + '__max')}",
                ]
            )
    stats_row = connection.execute("SELECT " + ", ".join(expressions) + " FROM data").fetchone()
    stats_columns = [item[0] for item in connection.description]
    stats = dict(zip(stats_columns, stats_row, strict=True))
    profiles: list[ColumnProfile] = []
    for name, data_type, *_ in selected:
        if data_type.startswith(("DATE", "TIMESTAMP")):
            kind = "date"
        elif NUMERIC_TYPES.match(data_type):
            kind = "measure"
        elif data_type.startswith(("VARCHAR", "BOOLEAN", "ENUM")):
            kind = "dimension"
        else:
            kind = "other"
        profiles.append(
            ColumnProfile(
                name=name,
                data_type=data_type,
                kind=kind,
                null_percent=float(stats[f"{name}__null"]),
                approximate_distinct=int(stats[f"{name}__distinct"]),
                minimum=json_safe(stats.get(f"{name}__min")),
                maximum=json_safe(stats.get(f"{name}__max")),
            )
        )
    sample_cursor = connection.execute("SELECT * FROM data USING SAMPLE 5 ROWS")
    sample_columns = [item[0] for item in sample_cursor.description]
    sample_rows = [
        {name: json_safe(value) for name, value in zip(sample_columns, row, strict=True)}
        for row in sample_cursor.fetchall()
    ]
    return DatasetProfile(
        row_count=row_count,
        columns=profiles,
        sample_rows=sample_rows,
        profile_is_partial=len(schema_rows) > max_columns,
    )


def friendly_duckdb_error(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0]
    if "CSV" in message.upper():
        return f"DuckDB could not read this CSV: {message}"
    if "PARQUET" in message.upper():
        return f"DuckDB could not read this Parquet file: {message}"
    return f"The dataset could not be registered: {message}"


def remove_dataset_files(paths: list[str], data_dir: Path) -> None:
    """Remove only resolved paths inside the configured data directory."""
    root = data_dir.resolve()
    targets: set[Path] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).resolve()
        if not path.is_relative_to(root):
            continue
        # Uploaded datasets and ML artifacts each live in a dedicated ID directory.
        targets.add(path if path.is_dir() else path.parent)
    for target in sorted(targets, key=lambda item: len(item.parts), reverse=True):
        if target == root or not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
