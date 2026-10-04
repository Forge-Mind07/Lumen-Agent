from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateDataset(Arguments):
    query: str = Field(default="SELECT * FROM data", min_length=1, max_length=20000)
    name: str = Field(default="Model dataset", max_length=120)


class InspectDataset(Arguments):
    dataset_id: str | None = None
    target: str | None = None


class TrainModel(Arguments):
    dataset_id: str | None = None
    task: Literal["classification", "regression"] | None = None
    target: str | None = None
    feature_columns: list[str] | None = Field(default=None, max_length=80)
    excluded_columns: list[str] | None = Field(default=None, max_length=100)
    metric: Literal["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "mae", "rmse", "r2"] | None = None
    algorithms: list[Literal["logistic_regression", "random_forest", "ridge"]] | None = Field(default=None, min_length=1, max_length=3)
    time_column: str | None = None
    positive_class: str | None = None
    test_size: float = Field(default=0.2, ge=0.15, le=0.35)
    random_seed: int = Field(default=42, ge=0, le=2147483647)


class ModelReference(Arguments):
    model_id: str | None = None


class CompareModels(Arguments):
    model_ids: list[str] | None = Field(default=None, max_length=10)


class Predict(ModelReference):
    dataset_id: str | None = None
    row: dict[str, Any] | None = None
    minimum_probability: float | None = Field(default=None, ge=0, le=1)
    limit: int = Field(default=100, ge=1, le=500)


class Evaluate(ModelReference):
    dataset_id: str | None = None


class ExplainPrediction(ModelReference):
    row: dict[str, Any]


class Cluster(Arguments):
    dataset_id: str | None = None
    feature_columns: list[str] | None = Field(default=None, max_length=80)
    excluded_columns: list[str] = Field(default_factory=list, max_length=100)
    clusters: int | None = Field(default=None, ge=2, le=8)
    random_seed: int = Field(default=42, ge=0, le=2147483647)


class Forecast(Arguments):
    dataset_id: str | None = None
    time_column: str
    value_column: str
    frequency: Literal["D", "W", "MS"] = "MS"
    aggregation: Literal["sum", "mean"] = "sum"
    horizon: int = Field(default=3, ge=1, le=52)


class Anomalies(Arguments):
    dataset_id: str | None = None
    feature_columns: list[str] | None = Field(default=None, max_length=80)
    excluded_columns: list[str] = Field(default_factory=list, max_length=100)
    contamination: float = Field(default=0.05, ge=0.001, le=0.2)
    time_column: str | None = None
    random_seed: int = Field(default=42, ge=0, le=2147483647)


TOOL_MODELS = {
    "create_dataset": (CreateDataset, "Materialize a bounded immutable SQL dataset snapshot for ML. Returns a reference, schema, provenance and fingerprint; never all rows."),
    "inspect_dataset": (InspectDataset, "Profile an ML dataset, infer task candidates, inspect target distribution and possible data-quality or leakage issues."),
    "train_model": (TrainModel, "Train and compare classification or regression pipelines. Omitted fields reuse active project state; excluded_columns replaces prior exclusions. Includes a baseline, validation selection and separate held-out test metrics. Models and preprocessing are saved automatically."),
    "compare_models": (CompareModels, "Compare saved models in this conversation. State when datasets or evaluation splits differ."),
    "list_project_models": (CompareModels, "List saved models and the active model in this project."),
    "predict": (Predict, "Apply a saved pipeline to a dataset reference or individual row. Returns bounded preview, probabilities, distribution, and a queryable prediction dataset reference for SQL filtering."),
    "evaluate_model": (Evaluate, "Evaluate a saved pipeline on a labeled dataset, or inspect its stored holdout evaluation."),
    "explain_model": (ModelReference, "Return measured held-out permutation importance and model limitations."),
    "explain_prediction": (ExplainPrediction, "Explain an individual prediction with feature replacement sensitivity, explicitly not causal or SHAP attribution."),
    "cluster_dataset": (Cluster, "Compare a small range of KMeans cluster counts and return segment sizes, feature profiles, and member predictions."),
    "forecast_series": (Forecast, "Aggregate a time series and compare naive and linear-trend forecasts using rolling-origin backtesting. Returns calibrated approximate intervals and frequency-gap warnings."),
    "detect_anomalies": (Anomalies, "Fit Isolation Forest and return anomaly scores and flagged rows. Anomalies are unusual observations, not confirmed fraud."),
}

ML_TOOLS = [{"type": "function", "function": {"name": name, "description": desc, "parameters": cls.model_json_schema()}} for name, (cls, desc) in TOOL_MODELS.items()]

ML_INSTRUCTIONS = """You are ML Agent. You also have read-only SQL and chart tools.
The backend runtime performs ALL training, evaluation, predictions and numerical calculations.
Never invent metrics or claim training has happened without successful tool results.
Begin with schema/SQL exploration if needed, then create_dataset and inspect_dataset before training.
Use immutable dataset references, never paste entire data into a message. run_sql may use an ML dataset_id, including prediction datasets, with the same table named data.
For a clear target, infer classification vs regression using its meaning and profile; ask only if ambiguity prevents a valid workflow. Never invent a definition for 'good customers' or similar objectives.
Default candidate selection is automatic. Honour exclusions, requested algorithms and optimization metric.
train_model already returns held-out evaluation and measured importance. Do not immediately repeat evaluate_model or explain_model unless the user requests additional evaluation or information is missing.
The active project below is durable state. Follow-ups like remove age and retrain must merge exclusions with the previous list; use train_model with appropriate overrides. Reuse dataset/model IDs from project state.
State which class is positive when discussing recall/probability. Explain the baseline comparison, held-out metrics, and leakage warnings. Completion is not evidence a model is useful.
Explain that performance is an estimate on this split and does not prove future effectiveness.
For risk filtering, use predict minimum_probability or query its prediction dataset with run_sql dataset_id; prediction_probability always refers to the documented positive class.
For false negatives, query the saved evaluation_dataset_id from train_model with SQL.
Do not write to user databases. Saving local model artifacts is part of the workflow.
Return a concise direct answer with 2–4 observations and optional next steps. Controlled result cards already show charts and metrics; do not duplicate them in giant markdown tables.
Keep the final reply under 180 words: one direct opening sentence, 2–4 observations, and at most two plain-language next steps. Do not repeat the metric table or expose tool names/model IDs in prose. Logistic regression is classification-only; ridge is regression-only. Suggest only algorithms supported for the active task.
Do not fabricate business meanings for clusters: use measured feature differences, and describe names as descriptive labels. Individual explanations are sensitivity checks, not causal conclusions.
"""
