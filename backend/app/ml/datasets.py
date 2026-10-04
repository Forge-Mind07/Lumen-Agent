from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from ..config import Settings
from ..models import new_id, utc_now
from ..sql_tool import execute_read_only_query
from ..storage import Storage


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in {"path", "duckdb_path", "artifact_path", "session_id"}}


class DatasetService:
    def __init__(self, storage: Storage, settings: Settings, session_id: str):
        self.storage, self.settings, self.session_id = storage, settings, session_id

    def get(self, dataset_id: str) -> dict[str, Any]:
        return self.storage.get_ml("datasets", dataset_id, self.session_id)

    def materialize(self, query: str, name: str) -> dict[str, Any]:
        session = self.storage.get_session(self.session_id)
        source = self.storage.get_dataset(session.dataset_id)
        result = execute_read_only_query(source.duckdb_path, query, row_limit=self.settings.ml_max_rows,
                                         timeout_seconds=self.settings.query_timeout_seconds)
        if result.truncated:
            raise ValueError(f"Dataset exceeds {self.settings.ml_max_rows:,} rows. Aggregate or select a representative bounded sample with SQL first.")
        query_artifact = self.storage.add_artifact(self.session_id, "query_result", {
            "sql": query, **result.as_payload(), "rows": result.rows[:5],
            "returned_row_count": min(5, len(result.rows)), "truncated": len(result.rows) > 5,
        })
        return self.save(result.columns, result.column_types, result.rows, name=name,
                         source_sql=query, source_dataset_id=source.id, query_id=query_artifact.id)

    def save(self, columns: list[str], types: list[str], rows: list[list[Any]], **provenance: Any) -> dict[str, Any]:
        if not rows:
            raise ValueError("The dataset is empty. Adjust the query before modeling.")
        column_limit = self.settings.ml_max_columns + (24 if provenance.get("source_model_id") else 0)
        if len(columns) > column_limit or len(set(columns)) != len(columns):
            raise ValueError(f"Use unique column names and at most {self.settings.ml_max_columns} columns.")
        if len(rows) > self.settings.ml_max_rows:
            raise ValueError("Prediction or dataset row limit exceeded")
        payload = {"columns": columns, "column_types": types, "rows": rows}
        encoded = json.dumps(payload, default=str, allow_nan=False, separators=(",", ":")).encode()
        if len(encoded) > self.settings.ml_max_snapshot_bytes:
            raise ValueError("Dataset is too large in memory. Select fewer columns or shorter text fields.")
        dataset_id = new_id("mlds")
        directory = (self.settings.data_dir / "ml" / "datasets" / dataset_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "snapshot.json"
        path.write_bytes(encoded)
        # The SQL executor is reused for every snapshot and prediction dataset.
        # Build a physical DuckDB table from a locally controlled dataframe.
        import pandas as pd
        frame = pd.DataFrame(rows, columns=columns)
        db_path = directory / "dataset.duckdb"
        with duckdb.connect(str(db_path)) as db:
            db.register("snapshot_frame", frame)
            db.execute("CREATE TABLE data AS SELECT * FROM snapshot_frame")
        record = {"id": dataset_id, "dataset_id": dataset_id, "session_id": self.session_id,
                  "path": str(path), "duckdb_path": str(db_path), "rows": len(rows),
                  "columns": [{"name": c, "type": t} for c, t in zip(columns, types)],
                  "fingerprint": hashlib.sha256(encoded).hexdigest(), "created_at": utc_now(),
                  "tables_used": ["data"], **provenance}
        return self.storage.put_ml("datasets", record)


def load_frame(path: str):
    import numpy as np
    import pandas as pd
    payload = json.loads(Path(path).read_text())
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame
