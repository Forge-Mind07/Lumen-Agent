from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.pipeline import Pipeline

from .preprocessing import normalize, preprocessor, select_features


def patterns(frame, args, directory: Path, progress, task):
    if len(frame) < 20:
        raise ValueError("At least 20 rows are needed to estimate meaningful patterns.")
    features, excluded, warnings = select_features(frame, requested=args.get("feature_columns"), excluded=args.get("excluded_columns"), time_column=args.get("time_column"))
    if not features:
        raise ValueError("No usable features remain. Select non-identifier features with variation.")
    prep, schema = preprocessor(frame[features])
    x = normalize(frame, schema)
    transformed = prep.fit_transform(x)
    seed = args["random_seed"]
    progress("training", "Comparing segments" if task == "clustering" else "Finding unusual observations")
    candidates, charts = [], []
    if task == "clustering":
        unique = len(np.unique(transformed, axis=0))
        counts = [args["clusters"]] if args.get("clusters") else list(range(2, min(6, unique - 1, len(frame)-1) + 1))
        for count in counts:
            if count >= unique:
                continue
            estimator = KMeans(n_clusters=count, n_init=5, random_state=seed)
            labels = estimator.fit_predict(transformed)
            score = float(silhouette_score(transformed, labels, sample_size=min(1000, len(frame)), random_state=seed))
            candidates.append({"algorithm": f"kmeans_{count}", "status": "completed", "validation_metrics": {"silhouette": score}, "estimator": estimator})
        if not candidates:
            raise ValueError("Not enough distinct observations for the requested segment count.")
        best = max(candidates, key=lambda item: item["validation_metrics"]["silhouette"])
        estimator = best["estimator"]
        labels = estimator.predict(transformed)
        profiles = []
        numeric = [c for c in features if schema[c] == "numeric"]
        for label in sorted(set(labels)):
            members = frame[labels == label]
            differences = []
            for col in numeric:
                std = frame[col].std()
                if std and np.isfinite(std):
                    differences.append((col, float((members[col].mean()-frame[col].mean())/std)))
            differences.sort(key=lambda v: abs(v[1]), reverse=True)
            name = ", ".join(("Higher " if delta > 0 else "Lower ") + col.replace("_", " ") for col, delta in differences[:2]) or f"Segment {int(label)+1}"
            profiles.append({"cluster": int(label), "name": name, "size": len(members), "share": len(members)/len(frame),
                             "means": {c: float(members[c].mean()) for c in numeric[:12]}})
        projection = PCA(n_components=min(2, transformed.shape[1]), random_state=seed).fit_transform(transformed)
        points = [{"x": float(projection[i, 0]), "y": float(projection[i, 1]) if projection.shape[1] > 1 else 0, "segment": int(labels[i])} for i in np.linspace(0, len(frame)-1, min(300, len(frame)), dtype=int)]
        charts = [{"type": "cluster_scatter", "title": "Segments · two-dimensional projection", "data": points},
                  {"type": "cluster_size", "title": "Segment sizes", "data": [{"label": p["name"], "value": p["size"]} for p in profiles]}]
        metrics = {"silhouette": best["validation_metrics"]["silhouette"], "segments": len(profiles)}
        predictions = frame.assign(segment=[profiles[int(v)]["name"] for v in labels], cluster=labels)
        algorithm = "kmeans"
        warnings.append("Segment labels describe measured differences, not verified business categories. Silhouette is an internal comparison, not a prediction accuracy score.")
        extra = {"cluster_profiles": profiles}
    else:
        estimator = IsolationForest(n_estimators=100, contamination=args["contamination"], n_jobs=1, random_state=seed)
        labels = estimator.fit_predict(transformed)
        scores = -estimator.score_samples(transformed)
        predictions = frame.assign(anomaly_score=scores, is_anomaly=labels == -1)
        metrics = {"flagged_rows": int((labels == -1).sum()), "flagged_share": float((labels == -1).mean())}
        hist, edges = np.histogram(scores, bins=16)
        charts = [{"type": "anomaly_distribution", "title": "Anomaly scores", "data": [{"label": round(float(edges[i]), 3), "value": int(v)} for i, v in enumerate(hist)]}]
        if args.get("time_column"):
            time_col = args["time_column"]
            if time_col not in frame:
                raise ValueError("Time column does not exist")
            timeline = predictions.assign(_time=pd.to_datetime(predictions[time_col], errors="coerce", utc=True)).dropna(subset=["_time"]).sort_values("_time")
            timeline = timeline.iloc[np.linspace(0, len(timeline)-1, min(300, len(timeline)), dtype=int)] if len(timeline) else timeline
            charts.append({"type": "anomaly_timeline", "title": "Unusual activity over time", "data": [{"x": str(r['_time']), "y": float(r['anomaly_score']), "flagged": bool(r['is_anomaly'])} for r in timeline.to_dict("records")]})
        algorithm, extra = "isolation_forest", {}
        warnings.append("Flagged rows are unusual, not confirmed fraud or errors. The contamination setting controls the estimated anomaly share.")
    progress("evaluating", "Summarizing measured patterns")
    pipeline = Pipeline([("preprocess", prep), ("model", estimator)])
    artifact = directory / "pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "schema": schema, "task": task, **extra}, artifact)
    predictions.to_json(directory / "predictions.json", orient="split", index=False)
    return {"task": task, "algorithm": algorithm, "metrics": metrics, "feature_schema": schema,
            "features": features, "excluded_features": excluded, "warnings": warnings, "charts": charts,
            "candidates": [{k: v for k, v in c.items() if k != "estimator"} for c in candidates],
            "artifact_path": str(artifact), "preprocessing": "Median imputation, standard scaling, and bounded one-hot encoding fitted and persisted with the model.",
            "model_configuration": {"random_seed": seed, "clusters": int(estimator.n_clusters)} if task == "clustering" else {"random_seed": seed, "contamination": args["contamination"]},
            "summary": "Patterns measured from the selected features.", **extra}


def forecast(frame, args, directory: Path, progress):
    time_col, value_col = args["time_column"], args["value_column"]
    if time_col not in frame or value_col not in frame:
        raise ValueError("Choose existing time and numeric value columns.")
    dates = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if dates.isna().any() or values.isna().any():
        raise ValueError("Forecasting requires complete valid dates and numeric values; clean them with SQL first.")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()
    series = series.resample(args["frequency"]).sum(min_count=1) if args["aggregation"] == "sum" else series.resample(args["frequency"]).mean()
    if series.isna().any():
        raise ValueError("The aggregated series has missing periods. Decide whether these mean zero or missing observations and fill explicitly with SQL.")
    n, horizon = len(series), args["horizon"]
    if n < max(12, horizon*2+6):
        raise ValueError("Not enough time periods for rolling backtests at this horizon. Supply more history or shorten the forecast.")
    season = {"D": 7, "W": 52, "MS": 12}[args["frequency"]]
    def estimate(name, history, steps):
        if name == "last_value":
            return np.repeat(history[-1], steps)
        if name == "seasonal_naive":
            return np.asarray([history[-season + i % season] for i in range(steps)])
        coefficient = np.polyfit(np.arange(len(history)), history, 1)
        return np.polyval(coefficient, np.arange(len(history), len(history)+steps))
    names = ["last_value", "linear_trend"]
    origins = sorted(origin for origin in (n-i*horizon for i in range(1, 5)) if origin >= 6)
    if min(origins) >= season*2:
        names.append("seasonal_naive")
    progress("training", "Comparing forecasts with rolling backtests")
    errors, candidates, folds = {}, [], []
    values = series.to_numpy(dtype=float)
    # The final horizon remains unseen during model selection.
    selection_origins = origins[:-1]
    if not selection_origins:
        raise ValueError("More history is needed for separate validation and test forecast windows.")
    for name in names:
        actual, predicted = [], []
        for origin in selection_origins:
            actual.extend(values[origin:origin+horizon])
            predicted.extend(estimate(name, values[:origin], horizon))
        errors[name] = np.asarray(actual)-np.asarray(predicted)
        candidates.append({"algorithm": name, "status": "completed", "validation_metrics": {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}})
    best = min(candidates, key=lambda c: c["validation_metrics"]["mae"])["algorithm"]
    test_origin = origins[-1]
    test_prediction = estimate(best, values[:test_origin], horizon)
    actual = values[test_origin:test_origin+horizon]
    metrics = {"mae": float(mean_absolute_error(actual, test_prediction)), "rmse": float(np.sqrt(mean_squared_error(actual, test_prediction)))}
    future = estimate(best, values, horizon)
    radius = float(np.quantile(abs(errors[best]), .9))
    future_dates = pd.date_range(series.index[-1], periods=horizon+1, freq=args["frequency"])[1:]
    output = pd.DataFrame({"date": future_dates.astype(str), "prediction": future, "lower": future-radius, "upper": future+radius})
    output.to_json(directory / "predictions.json", orient="split", index=False)
    points = [{"date": str(d), "actual": float(v)} for d, v in series.iloc[-150:].items()] + output.to_dict("records")
    artifact = directory / "pipeline.joblib"
    joblib.dump({"task": "forecasting", "algorithm": best, "history": values, "configuration": args}, artifact)
    progress("evaluating", "Measuring forecast error and approximate ranges")
    return {"task": "forecasting", "algorithm": best, "metrics": metrics, "candidates": candidates,
            "forecast": output.to_dict("records"), "charts": [{"type": "forecast", "title": f"{value_col.replace('_', ' ')} forecast", "data": points}],
            "features": [time_col], "target": value_col, "artifact_path": str(artifact), "model_configuration": args,
            "split": {"strategy": "rolling-origin validation and final horizon holdout", "backtest_origins": selection_origins, "test_start": str(series.index[test_origin]), "test_rows": horizon},
            "warnings": ["Ranges use the 90th percentile of absolute validation errors and are approximate, not guaranteed coverage.", "Simple univariate forecasts do not account for holidays, promotions or structural changes. Small backtests give uncertain estimates."],
            "summary": "Compared simple forecasting methods on earlier periods and evaluated the winner on the final unseen horizon."}
