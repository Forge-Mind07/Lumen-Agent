from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import normalize, preprocessor, select_features


def metrics_for(task: str, truth, prediction, probabilities=None, classes=None, positive=None):
    if task == "regression":
        return {"mae": float(mean_absolute_error(truth, prediction)), "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
                "r2": float(r2_score(truth, prediction))}
    binary = len(classes) == 2
    kwargs = {"average": "binary", "pos_label": positive} if binary else {"average": "macro"}
    metrics = {"accuracy": float(accuracy_score(truth, prediction)),
               "precision": float(precision_score(truth, prediction, zero_division=0, **kwargs)),
               "recall": float(recall_score(truth, prediction, zero_division=0, **kwargs)),
               "f1": float(f1_score(truth, prediction, zero_division=0, **kwargs))}
    if binary and probabilities is not None and len(set(truth)) == 2:
        scores = probabilities[:, list(classes).index(positive)]
        y = np.asarray(truth) == positive
        metrics.update(roc_auc=float(roc_auc_score(y, scores)), pr_auc=float(average_precision_score(y, scores)))
    return metrics


def performance_charts(task, truth, prediction, probabilities, classes, positive):
    if task == "regression":
        points = [{"actual": float(a), "predicted": float(b), "residual": float(a-b)} for a, b in zip(truth[:200], prediction[:200])]
        hist, edges = np.histogram(np.asarray(truth) - prediction, bins=16)
        return [{"type": "actual_vs_predicted", "title": "Predictions on held-out data", "data": points},
                {"type": "residual_distribution", "title": "Prediction errors", "data": [{"label": round(float(edges[i]), 3), "value": int(v)} for i, v in enumerate(hist)]}]
    matrix = confusion_matrix(truth, prediction, labels=classes)
    charts = [{"type": "confusion_matrix", "title": "Held-out predictions", "labels": list(classes), "matrix": matrix.tolist()}]
    if len(classes) == 2 and probabilities is not None and len(set(truth)) == 2:
        scores = probabilities[:, list(classes).index(positive)]
        y = np.asarray(truth) == positive
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        for name, title, xs, ys in [("roc_curve", "ROC curve", fpr, tpr), ("precision_recall_curve", "Precision and recall", recall, precision)]:
            indexes = np.unique(np.linspace(0, len(xs)-1, min(120, len(xs)), dtype=int))
            charts.append({"type": name, "title": title, "data": [{"x": float(xs[i]), "y": float(ys[i])} for i in indexes]})
    return charts


def train(frame: pd.DataFrame, args: dict, directory: Path, progress: Callable) -> dict:
    target, task, seed = args.get("target"), args.get("task"), args["random_seed"]
    if not target or target not in frame:
        raise ValueError("Choose an existing target column before training.")
    if task not in {"classification", "regression"}:
        raise ValueError("Specify classification or regression based on the target's meaning.")
    dropped = int(frame[target].isna().sum())
    original_length = len(frame)
    data = frame.dropna(subset=[target]).drop_duplicates().copy()
    if len(data) < 40:
        raise ValueError("At least 40 complete, distinct labeled rows are needed for train, validation and test splits.")
    warnings = [f"Removed {dropped} rows with missing targets and {original_length - dropped - len(data)} exact duplicate rows."]
    if task == "classification":
        data[target] = data[target].astype(str)
        counts = data[target].value_counts()
        if len(counts) < 2:
            raise ValueError("Classification needs at least two target classes.")
        if len(counts) > 20 or counts.min() < 5:
            raise ValueError("Use at most 20 classes with at least 5 examples of each class; gather more examples or revise the target.")
        positive = args.get("positive_class") or sorted(counts.index)[-1]
        if positive not in counts:
            raise ValueError("The positive class is not present in the target.")
    else:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        if data[target].isna().any() or not np.isfinite(data[target]).all() or data[target].nunique() < 2:
            raise ValueError("Regression needs a finite numeric target with at least two distinct values.")
        positive = None
    time_column = args.get("time_column")
    if time_column:
        if time_column not in data:
            raise ValueError("Time column does not exist.")
        dates = pd.to_datetime(data[time_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("Time-aware splits require a valid timestamp in every row.")
        data = data.assign(_split_time=dates).sort_values("_split_time").drop(columns="_split_time")
        # Never put the same timestamp in different partitions.
        times = np.sort(dates.unique())
        if len(times) < 10:
            raise ValueError("Time-aware training needs at least 10 distinct timestamps.")
        test_start = times[max(1, int(len(times)*(1-args["test_size"])))]
        val_start = times[max(1, int(len(times)*(1-args["test_size"])*.8))]
        ordered_dates = pd.to_datetime(data[time_column], utc=True)
        train_data = data[ordered_dates < val_start]
        val_data = data[(ordered_dates >= val_start) & (ordered_dates < test_start)]
        test_data = data[ordered_dates >= test_start]
        split_strategy = "chronological, equal timestamps kept together"
    else:
        stratify = data[target] if task == "classification" else None
        development, test_data = train_test_split(data, test_size=args["test_size"], random_state=seed, stratify=stratify)
        train_data, val_data = train_test_split(development, test_size=.25, random_state=seed,
                                               stratify=development[target] if task == "classification" else None)
        split_strategy = "stratified_random" if task == "classification" else "random"
    if min(len(train_data), len(val_data), len(test_data)) < 4:
        raise ValueError("Insufficient rows in one split; increase dataset size or change the split strategy.")
    if task == "classification" and any(set(part[target]) != set(data[target]) for part in [train_data, val_data, test_data]):
        raise ValueError("Each split must contain every class; use more labeled rows or a different time window.")
    progress("checking", "Checking data quality and leakage")
    features, excluded, notices = select_features(train_data, target, args.get("feature_columns"), args.get("excluded_columns"), time_column)
    warnings.extend(notices)
    if not features:
        raise ValueError("No usable features remain after identifier, missing-data and leakage checks.")
    if not time_column:
        warnings.append("Random splits assume independent rows. For repeated customers or temporal outcomes, materialize one row per entity or specify a time column.")
    prep, schema = preprocessor(train_data[features])
    x_train, x_val, x_test = (normalize(part, schema) for part in (train_data, val_data, test_data))
    y_train, y_val, y_test = (part[target] for part in (train_data, val_data, test_data))
    if task == "classification":
        algorithms = {"logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
                      "random_forest": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed)}
        baseline = DummyClassifier(strategy="prior")
        metric = args.get("metric") or ("roc_auc" if data[target].nunique() == 2 else "f1")
        allowed = {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"}
    else:
        algorithms = {"ridge": Ridge(alpha=1.0), "random_forest": RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=seed)}
        baseline = DummyRegressor(strategy="mean")
        metric = args.get("metric") or "rmse"
        allowed = {"mae", "rmse", "r2"}
    if metric not in allowed or (task == "classification" and data[target].nunique() != 2 and metric in {"roc_auc", "pr_auc"}):
        raise ValueError("The optimization metric is not supported for this task or target.")
    names = list(dict.fromkeys(args.get("algorithms") or algorithms.keys()))
    if not set(names) <= set(algorithms):
        raise ValueError("Requested algorithms do not support this task.")
    candidates = [("baseline", baseline)] + [(name, algorithms[name]) for name in names]
    results, fitted = [], {}
    for name, estimator in candidates:
        progress("training", f"Training {name.replace('_', ' ')}")
        started = time.perf_counter()
        # Each candidate gets its own fitted preprocessing pipeline.
        from sklearn.base import clone
        pipeline = Pipeline([("preprocess", clone(prep)), ("model", estimator)])
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_val)
            probabilities = pipeline.predict_proba(x_val) if task == "classification" else None
            classes = pipeline.classes_ if task == "classification" else None
            measured = metrics_for(task, y_val, predictions, probabilities, classes, positive)
            results.append({"algorithm": name, "status": "completed", "validation_metrics": measured,
                            "duration_ms": round((time.perf_counter()-started)*1000)})
            fitted[name] = pipeline
        except Exception:
            results.append({"algorithm": name, "status": "failed", "error": "Candidate could not fit this dataset."})
    successful = [r for r in results if r["status"] == "completed" and metric in r["validation_metrics"]]
    if not successful:
        raise ValueError("All model candidates failed; review feature types and target values.")
    direction = -1 if metric in {"mae", "rmse"} else 1
    best = max(successful, key=lambda r: direction*r["validation_metrics"][metric])
    pipeline = fitted[best["algorithm"]]
    development = pd.concat([train_data, val_data])
    pipeline.fit(normalize(development, schema), development[target])
    progress("evaluating", "Evaluating on the untouched test split")
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test) if task == "classification" else None
    classes = pipeline.classes_ if task == "classification" else None
    metrics = metrics_for(task, y_test, predictions, probabilities, classes, positive)
    baseline_pipeline = fitted.get("baseline")
    baseline_metrics = {}
    if baseline_pipeline is not None:
        baseline_pipeline.fit(normalize(development, schema), development[target])
        baseline_metrics = metrics_for(task, y_test, baseline_pipeline.predict(x_test),
                                      baseline_pipeline.predict_proba(x_test) if task == "classification" else None,
                                      baseline_pipeline.classes_ if task == "classification" else None, positive)
    progress("explaining", "Measuring feature importance")
    # Original-column permutation importance, bounded to a reproducible holdout sample.
    sample = x_test.sample(min(500, len(x_test)), random_state=seed)
    scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
    importance = permutation_importance(pipeline, sample, y_test.loc[sample.index], n_repeats=3, random_state=seed, scoring=scoring, n_jobs=1)
    influences = sorted([{"feature": c, "importance": float(v), "std": float(s)} for c, v, s in zip(features, importance.importances_mean, importance.importances_std)], key=lambda r: r["importance"], reverse=True)
    charts = performance_charts(task, y_test.to_numpy(), predictions, probabilities, classes, positive)
    charts.append({"type": "feature_importance", "title": "Most influential signals", "data": influences[:12]})
    if baseline_metrics and direction*metrics[metric] <= direction*baseline_metrics[metric]:
        warnings.append("This model did not outperform the simple baseline on the held-out test metric. Do not rely on it without further investigation.")
    warnings.append("Repeated model adjustments can overfit this test set. Confirm final performance on fresh data before operational use.")
    reference = {c: (float(development[c].median()) if schema[c] == "numeric" else str(development[c].mode().iloc[0]) if not development[c].mode().empty else None) for c in features}
    artifact = directory / "pipeline.joblib"
    bundle = {"pipeline": pipeline, "schema": schema, "task": task, "target": target, "positive_class": positive, "reference": reference}
    joblib.dump(bundle, artifact)
    evaluation = test_data.copy()
    evaluation["actual"] = y_test
    evaluation["prediction"] = predictions
    if probabilities is not None and len(classes) == 2:
        evaluation["prediction_probability"] = probabilities[:, list(classes).index(positive)]
        evaluation["false_negative"] = (y_test == positive) & (predictions != positive)
    evaluation.to_json(directory / "evaluation.json", orient="split", index=False)
    return {"task": task, "target": target, "algorithm": best["algorithm"], "metric": metric, "metrics": metrics,
            "baseline_metrics": baseline_metrics, "candidates": results, "feature_importance": influences,
            "features": features, "excluded_features": excluded, "feature_schema": schema,
            "positive_class": positive, "classes": list(classes) if classes is not None else None,
            "split": {"strategy": split_strategy, "train_rows": len(train_data), "validation_rows": len(val_data), "test_rows": len(test_data), "refit_rows": len(development)},
            "preprocessing": "Training-fitted median imputation + standard scaling; missing category imputation + one-hot encoding (24 categories/feature, unknowns ignored). Full pipeline persisted.",
            "model_configuration": {k: v for k, v in pipeline.named_steps["model"].get_params().items() if isinstance(v, (str, int, float, bool, type(None)))},
            "importance_method": f"Held-out permutation, 3 repeats, {scoring}; negative values retained; correlation is not causation.",
            "artifact_path": str(artifact), "charts": charts, "warnings": warnings,
            "summary": "Compared candidates on validation data and measured the selected pipeline on a separate test split."}
