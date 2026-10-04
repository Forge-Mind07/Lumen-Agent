from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def inspect(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    if target and target not in frame:
        raise ValueError(f"Target {target!r} is not a column in the dataset")
    columns = [{"name": str(c), "type": str(frame[c].dtype), "missing_percent": round(float(frame[c].isna().mean() * 100), 2),
                "distinct": int(frame[c].nunique())} for c in frame]
    warnings = []
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows found; supervised training removes exact duplicates before splitting.")
    _, excluded, feature_warnings = select_features(frame, target)
    warnings.extend(feature_warnings)
    values = frame[target].dropna() if target else None
    distribution = ({str(k): int(v) for k, v in values.value_counts().head(20).items()} if target else {})
    task = None if target is None else ("classification" if not pd.api.types.is_numeric_dtype(values) or values.nunique() <= 10 else "regression")
    return {"rows": len(frame), "columns": columns, "target": target, "target_distribution": distribution,
            "suggested_task": task, "task_note": "Confirm target meaning; low cardinality numeric targets can also be regression.",
            "excluded_candidates": excluded, "warnings": warnings}


def select_features(frame: pd.DataFrame, target: str | None = None, requested: list[str] | None = None,
                    excluded: list[str] | None = None, time_column: str | None = None):
    invalid = set((requested or []) + (excluded or [])) - set(frame.columns)
    if invalid:
        raise ValueError("Unknown features: " + ", ".join(sorted(invalid)))
    selected, reasons, warnings = [], {}, []
    y = frame[target] if target else None
    for name in requested if requested is not None else frame.columns:
        series = frame[name]
        reason = None
        if name == target:
            reason = "target"
        elif name in (excluded or []) or name == time_column:
            reason = "excluded by configuration"
        elif series.isna().mean() >= .95:
            reason = "at least 95% missing"
        elif series.nunique() <= 1:
            reason = "constant"
        elif re.search(r"(^id$|_id$|^id_|uuid|email|phone|postcode)", name, re.I):
            reason = "possible identifier or personal identifier"
        elif re.search(r"(after_|post_|future_|outcome|resolved_at|cancelled_at|churn_date|closed_at)", name, re.I):
            reason = "possible post-outcome information; verify availability at prediction time"
        elif target and series.astype(str).equals(y.astype(str)):
            reason = "duplicates the target (possible leakage)"
        elif target and 1 < series.nunique() == y.nunique() <= 20 and pd.crosstab(series, y).gt(0).sum(axis=1).eq(1).all():
            reason = "one-to-one encoding of the target (possible leakage)"
        elif target and pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y) and abs(series.corr(y)) > .995:
            reason = "near-perfect target correlation (possible leakage)"
        elif not pd.api.types.is_numeric_dtype(series) and series.nunique() > min(100, max(20, len(frame) * .8)):
            reason = "high cardinality text or identifier; engineer a smaller feature"
        elif not pd.api.types.is_numeric_dtype(series) and series.dropna().astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > .8:
            reason = "raw timestamp; engineer prediction-time-safe date features with SQL"
        if reason:
            reasons[name] = reason
            if name != target:
                warnings.append(f"Excluded {name}: {reason}.")
        else:
            selected.append(name)
    warnings.append("Automated leakage checks cannot establish when features became available. Confirm all features exist at prediction time.")
    return selected, reasons, warnings


def normalize(frame: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    missing = set(schema) - set(frame.columns)
    if missing:
        raise ValueError("Missing prediction features: " + ", ".join(sorted(missing)))
    data = frame[list(schema)].copy()
    for name, kind in schema.items():
        if kind == "numeric":
            values = pd.to_numeric(data[name], errors="coerce")
            if (data[name].notna() & values.isna()).any():
                raise ValueError(f"Feature {name!r} contains non-numeric values")
            data[name] = values.replace([np.inf, -np.inf], np.nan)
        else:
            data[name] = data[name].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return data


def preprocessor(frame: pd.DataFrame):
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame if c not in numeric]
    schema = {c: "numeric" if c in numeric else "categorical" for c in frame}
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="(missing)")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=24, sparse_output=False))]), categorical),
    ], sparse_threshold=0)
    return transformer, schema
