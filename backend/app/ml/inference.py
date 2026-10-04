from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import normalize
from .supervised import metrics_for, performance_charts


def infer(frame, args, model, directory: Path, operation):
    # Artifact locations only originate in the server-side scoped registry.
    bundle = joblib.load(model["artifact_path"])
    task = bundle["task"]
    if task == "forecasting":
        raise ValueError("Use forecast_series to build a new forecast with updated history.")
    pipeline, schema = bundle["pipeline"], bundle["schema"]
    x = normalize(frame, schema)
    prediction = pipeline.predict(x)
    probabilities = pipeline.predict_proba(x) if hasattr(pipeline, "predict_proba") else None
    positive = bundle.get("positive_class")
    if operation == "evaluate_model":
        target = bundle.get("target")
        if task not in {"classification", "regression"} or target not in frame:
            raise ValueError("Evaluation requires the trained target column in a labeled classification/regression dataset.")
        truth = frame[target]
        if truth.isna().any():
            raise ValueError("Evaluation target contains missing values.")
        truth = truth.astype(str) if task == "classification" else pd.to_numeric(truth, errors="raise")
        classes = pipeline.classes_ if task == "classification" else None
        if classes is not None and not set(truth) <= set(classes):
            raise ValueError("Evaluation data contains target classes unseen during training.")
        measured = metrics_for(task, truth, prediction, probabilities, classes, positive)
        return {"task": task, "model_id": model["id"], "metrics": measured,
                "charts": performance_charts(task, truth.to_numpy(), prediction, probabilities, classes, positive),
                "summary": "Evaluated the saved pipeline on the specified labeled dataset.",
                "warnings": ["This is an external evaluation only if these rows were not used in training or model selection."]}
    if operation == "explain_prediction":
        reference = bundle.get("reference")
        if not reference or len(frame) != 1:
            raise ValueError("Individual sensitivity explanations require one row and a supervised model.")
        def score(row):
            if probabilities is not None:
                if len(pipeline.classes_) != 2:
                    raise ValueError("Individual sensitivity currently supports binary classification and regression.")
                return float(pipeline.predict_proba(row)[0, list(pipeline.classes_).index(positive)])
            return float(pipeline.predict(row)[0])
        base = score(x)
        effects = []
        for feature in schema:
            altered = x.copy()
            altered[feature] = reference[feature]
            effects.append({"feature": feature, "importance": base-score(altered)})
        effects.sort(key=lambda item: abs(item["importance"]), reverse=True)
        return {"task": task, "model_id": model["id"], "prediction": str(prediction[0]), "score": base,
                "feature_importance": effects, "charts": [{"type": "feature_importance", "title": "Prediction sensitivity", "data": effects[:12]}],
                "summary": "Measured changes when each feature is replaced with its training median or most common category.",
                "warnings": ["Sensitivity is not causal attribution or SHAP; replacement values can create unrealistic feature combinations."]}
    reserved = {"prediction", "prediction_probability", "anomaly_score", "is_anomaly", "segment"}
    collisions = reserved & set(frame.columns)
    if collisions:
        raise ValueError("Rename existing output columns before prediction: " + ", ".join(sorted(collisions)))
    output = frame.copy()
    output["prediction"] = prediction
    charts, distribution = [], []
    if probabilities is not None:
        classes = list(pipeline.classes_)
        for index, label in enumerate(classes):
            probability_col = f"probability_class_{index}"
            if probability_col in output:
                raise ValueError(f"Rename existing {probability_col} before prediction")
            output[probability_col] = probabilities[:, index]
        if len(classes) == 2:
            output["prediction_probability"] = probabilities[:, classes.index(positive)]
            scores = output["prediction_probability"]
            distribution = [{"label": "Below 40%", "value": int((scores < .4).sum())},
                            {"label": "40–80%", "value": int(((scores >= .4) & (scores < .8)).sum())},
                            {"label": "80% and above", "value": int((scores >= .8).sum())}]
            charts.append({"type": "prediction_distribution", "title": f"Probability of {positive}", "data": distribution})
    if task == "anomaly_detection":
        output["anomaly_score"] = -pipeline.score_samples(x)
        output["is_anomaly"] = prediction == -1
        output = output.sort_values("anomaly_score", ascending=False)
    if task == "clustering":
        profiles = bundle["cluster_profiles"]
        output["segment"] = [profiles[int(v)]["name"] for v in prediction]
    if args.get("minimum_probability") is not None:
        if "prediction_probability" not in output:
            raise ValueError("Probability filtering requires a binary classifier.")
        output = output[output.prediction_probability >= args["minimum_probability"]]
    if "prediction_probability" in output:
        output = output.sort_values("prediction_probability", ascending=False)
    output.to_json(directory / "predictions.json", orient="split", index=False)
    return {"task": task, "model_id": model["id"], "metrics": {"predicted_rows": len(frame), "matching_rows": len(output)},
            "charts": charts, "prediction_distribution": distribution, "positive_class": positive,
            "probability_columns": {f"probability_class_{i}": str(c) for i, c in enumerate(pipeline.classes_)} if probabilities is not None else {},
            "summary": "Applied the saved preprocessing and model to the selected rows.",
            "warnings": ["Probabilities are model estimates and may not be calibrated. Distribution counts describe all input rows before filtering."] if probabilities is not None else []}
