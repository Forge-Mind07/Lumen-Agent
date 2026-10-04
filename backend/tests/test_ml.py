import asyncio
import io
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.agent import execute_tool, run_analysis
from app.datasets import register_dataset
from app.main import create_app
from app.ml.contracts import TrainModel
from app.ml.datasets import DatasetService
from app.ml.preprocessing import normalize, select_features
from app.ml.service import MLService
from app.ml.supervised import train
from app.models import AssistantTurn, ToolCall
from app.storage import Storage


def sample_frame(n=160):
    rng = np.random.default_rng(42)
    usage = rng.normal(50, 15, n)
    age = rng.integers(18, 75, n)
    score = usage / 15 + (age > 45) + rng.normal(0, 1.5, n)
    return pd.DataFrame({"customer_id": np.arange(n), "usage": usage, "age": age,
                         "plan": np.where(age > 45, "premium", "basic"),
                         "null_heavy": [None]*n,
                         "churned": (score < np.median(score)).astype(int),
                         "value": usage*3 + rng.normal(0, 20, n)})


def setup_ml(settings, storage, frame=None):
    frame = sample_frame() if frame is None else frame
    data = register_dataset(filename="customers.csv", source=io.BytesIO(frame.to_csv(index=False).encode()), storage=storage, settings=settings)
    session = storage.create_session(data.id, "Predict churn", "ml")
    storage.add_message(session.id, "user", "Predict churn using churned as the target")
    return session, MLService(storage, settings, session.id)


async def call(service, name, arguments):
    updates = [u async for u in service.execute(name, arguments)]
    return next(u["result"] for u in reversed(updates) if u["type"] == "result")


@pytest.mark.asyncio
async def test_classification_registry_prediction_followup(settings, storage):
    session, service = setup_ml(settings, storage)
    ref = await call(service, "create_dataset", {"query": "SELECT * EXCLUDE (value) FROM data"})
    assert ref["rows"] == 160 and "path" not in ref and "sample_rows" not in ref
    profile = await call(service, "inspect_dataset", {"target": "churned"})
    assert profile["suggested_task"] == "classification"
    result = await call(service, "train_model", {"task": "classification", "target": "churned"})
    assert 0 <= result["metrics"]["roc_auc"] <= 1
    assert len(result["candidates"]) == 3
    assert result["split"]["test_rows"] == 32
    assert "customer_id" in result["excluded_features"] and "null_heavy" in result["excluded_features"]
    assert result["evaluation_dataset_id"]
    state = Storage(settings.sqlite_path)
    restored = MLService(state, settings, session.id)
    assert restored.project()["active_model_id"] == result["model_id"]
    prediction = await call(restored, "predict", {"minimum_probability": .5, "limit": 10})
    assert len(prediction["predictions"]["rows"]) <= 10
    assert prediction["prediction_dataset_id"]
    columns = prediction["predictions"]["columns"]
    assert all(r[columns.index("prediction_probability")] >= .5 for r in prediction["predictions"]["rows"])
    sql, _ = await execute_tool("run_sql", {"dataset_id": prediction["prediction_dataset_id"], "query": "SELECT count(*) AS n FROM data WHERE prediction_probability >= 0.5"}, session_id=session.id, duckdb_path="unused", storage=storage, settings=settings)
    assert sql["rows"][0][0] == prediction["metrics"]["matching_rows"]
    retrained = await call(restored, "train_model", {"excluded_columns": ["age"], "algorithms": ["logistic_regression"], "metric": "recall"})
    assert "age" not in retrained["features"]
    assert retrained["target"] == "churned" and retrained["model_id"] != result["model_id"]
    assert len((await call(restored, "compare_models", {}))["models"]) == 2
    row = {"usage": 22, "plan": "new_plan"}
    single = await call(restored, "predict", {"row": row})
    assert single["metrics"]["predicted_rows"] == 1
    explanation = await call(restored, "explain_prediction", {"row": row})
    assert explanation["feature_importance"]
    model = storage.get_ml("models", retrained["model_id"], session.id)
    bundle = joblib.load(model["artifact_path"])
    expected = bundle["pipeline"].predict(normalize(pd.DataFrame([row]), bundle["schema"]))[0]
    p = single["predictions"]
    assert str(p["rows"][0][p["columns"].index("prediction")]) == str(expected)
    with pytest.raises(ValueError, match="Missing prediction features"):
        await call(restored, "predict", {"row": {"usage": 30}})


@pytest.mark.asyncio
async def test_regression_and_external_evaluation(settings, storage):
    session, service = setup_ml(settings, storage)
    await call(service, "create_dataset", {"query": "SELECT * EXCLUDE (churned) FROM data"})
    result = await call(service, "train_model", {"task": "regression", "target": "value", "metric": "mae"})
    assert result["metrics"]["rmse"] >= 0
    assert result["metrics"]["r2"] > .5
    assert result["feature_importance"]
    evaluated = await call(service, "evaluate_model", {"dataset_id": result["evaluation_dataset_id"]})
    assert evaluated["metrics"]["mae"] == pytest.approx(result["metrics"]["mae"], rel=1e-6)
    assert (await call(service, "predict", {"row": {"usage": 40, "age": 35, "plan": "basic"}}))["predictions"]["returned_row_count"] == 1


@pytest.mark.asyncio
async def test_patterns_and_forecasting(settings, storage):
    _, service = setup_ml(settings, storage)
    await call(service, "create_dataset", {"query": "SELECT usage, age, plan FROM data"})
    clusters = await call(service, "cluster_dataset", {"clusters": 3})
    assert len(clusters["cluster_profiles"]) == 3
    assert sum(p["size"] for p in clusters["cluster_profiles"]) == 160
    assert (await call(service, "predict", {}))["metrics"]["predicted_rows"] == 160
    anomalies = await call(service, "detect_anomalies", {"contamination": .1})
    assert 0 < anomalies["metrics"]["flagged_rows"] < 160
    frame = pd.DataFrame({"month": pd.date_range("2020-01-01", periods=48, freq="MS").astype(str), "revenue": np.arange(48)*10 + np.sin(np.arange(48))*3+100})
    _, forecast_service = setup_ml(settings, storage, frame)
    await call(forecast_service, "create_dataset", {})
    result = await call(forecast_service, "forecast_series", {"time_column": "month", "value_column": "revenue", "horizon": 3})
    assert len(result["forecast"]) == 3
    assert all(r["lower"] <= r["prediction"] <= r["upper"] for r in result["forecast"])
    assert result["split"]["test_rows"] == 3
    assert result["algorithm"] == "linear_trend"


@pytest.mark.parametrize("change,match", [
    ("missing", "existing target"), ("small", "At least 40"), ("oneclass", "at least two"),
    ("nulltarget", "At least 40"), ("unknownfeature", "Unknown features"), ("wrongmetric", "optimization metric"),
    ("badalgorithm", "algorithms do not support"), ("nonnumeric", "finite numeric"),
])
def test_invalid_training_inputs(tmp_path, change, match):
    frame = sample_frame()
    args = TrainModel(task="classification", target="churned").model_dump()
    if change == "missing": args["target"] = "absent"
    if change == "small": frame = frame.head(10)
    if change == "oneclass": frame["churned"] = 1
    if change == "nulltarget": frame["churned"] = None
    if change == "unknownfeature": args["feature_columns"] = ["absent"]
    if change == "wrongmetric": args["metric"] = "rmse"
    if change == "badalgorithm": args["algorithms"] = ["ridge"]
    if change == "nonnumeric": args.update(task="regression", target="plan")
    with pytest.raises(ValueError, match=match):
        train(frame, args, tmp_path, lambda *a: None)


def test_leakage_and_null_heavy_features():
    frame = sample_frame()
    frame["target_copy"] = frame["churned"]
    frame["post_outcome_timestamp"] = np.arange(len(frame))
    frame["encoded_outcome"] = frame["churned"].map({0: "red", 1: "blue"})
    features, excluded, warnings = select_features(frame, "churned")
    assert not {"customer_id", "target_copy", "post_outcome_timestamp", "null_heavy", "encoded_outcome"} & set(features)
    assert "duplicates the target" in excluded["target_copy"]
    assert any("prediction time" in w for w in warnings)


def test_missing_features_reproducibility_and_candidate_failure(tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline
    frame = sample_frame()
    frame.loc[frame.index % 4 == 0, "usage"] = np.nan
    frame.loc[frame.index % 5 == 0, "plan"] = None
    args = TrainModel(task="classification", target="churned", excluded_columns=["value"], algorithms=["logistic_regression"]).model_dump()
    first = train(frame, args, tmp_path, lambda *a: None)
    second = train(frame, args, tmp_path, lambda *a: None)
    assert first["metrics"] == second["metrics"]
    assert {"usage", "plan"} <= set(first["features"])
    def fail(*args, **kwargs):
        raise RuntimeError("Internal stack trace should not escape")
    monkeypatch.setattr(Pipeline, "fit", fail)
    with pytest.raises(ValueError, match="All model candidates failed"):
        train(frame, args, tmp_path, lambda *a: None)


def test_chronological_split_and_duplicate_times(tmp_path):
    frame = sample_frame()
    frame["observed_at"] = pd.date_range("2020-01-01", periods=80, freq="D").repeat(2).astype(str)
    result = train(frame, TrainModel(task="regression", target="value", time_column="observed_at", algorithms=["ridge"]).model_dump(), tmp_path, lambda *a: None)
    assert result["split"]["strategy"].startswith("chronological")
    assert result["split"]["test_rows"] == 32
    assert "observed_at" not in result["features"]


@pytest.mark.asyncio
async def test_mode_isolation_and_scoped_references(settings, storage):
    session, service = setup_ml(settings, storage)
    ref = await call(service, "create_dataset", {})
    sql_session = storage.create_session(session.dataset_id, "SQL exploration")
    with pytest.raises(ValueError, match="only in ML"):
        MLService(storage, settings, sql_session.id)
    second = storage.create_session(session.dataset_id, "Other project", "ml")
    with pytest.raises(ValueError, match="not found"):
        DatasetService(storage, settings, second.id).get(ref["dataset_id"])
    with pytest.raises(ValueError, match="require ML"):
        await execute_tool("run_sql", {"query": "SELECT * FROM data", "dataset_id": ref["dataset_id"]}, session_id=sql_session.id, duckdb_path="unused", storage=storage, settings=settings)
    with pytest.raises(ValueError, match="Invalid tool arguments"):
        await call(service, "train_model", {"task": "made_up", "python": "print(1)"})
    with pytest.raises(ValueError, match="SELECT-style"):
        await call(service, "create_dataset", {"query": "DELETE FROM data"})
    settings.ml_max_rows = 100
    with pytest.raises(ValueError, match="exceeds"):
        await call(service, "create_dataset", {})


@pytest.mark.asyncio
async def test_failed_run_is_persisted(settings, storage):
    session, service = setup_ml(settings, storage)
    await call(service, "create_dataset", {})
    with pytest.raises(ValueError, match="existing target"):
        await call(service, "train_model", {"target": "absent", "task": "classification"})
    run = storage.list_ml("runs", session.id)[0]
    assert run["status"] == "failed" and "absent" not in run.get("result", {})
    assert service.project()["status"] == "failed"
    assert service.project()["active_model_id"] is None


@pytest.mark.asyncio
async def test_actual_progress_and_timeout(settings, storage):
    session, service = setup_ml(settings, storage)
    await call(service, "create_dataset", {})
    settings.ml_timeout_seconds = .001
    with pytest.raises(ValueError, match="time limit"):
        await call(service, "train_model", {"target": "churned", "task": "classification"})
    assert storage.list_ml("runs", session.id)[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_ml_chat_tool_routing_and_bounded_context(settings, storage):
    session, _ = setup_ml(settings, storage)
    class Provider:
        requests = []
        async def complete(self, messages, tools, role="primary"):
            self.requests.append((messages.copy(), tools))
            if len(self.requests) == 1:
                return AssistantTurn(tool_calls=[ToolCall(id="materialize", name="create_dataset", arguments={"query": "SELECT * FROM data"})])
            return AssistantTurn(content="The dataset is ready for modeling.")
    provider = Provider()
    events = [e async for e in run_analysis(session_id=session.id, storage=storage, provider=provider, settings=settings)]
    assert "ml_result" in [e.type for e in events]
    assert "ML Agent" in provider.requests[0][0][0]["content"]
    assert "train_model" in [t["function"]["name"] for t in provider.requests[0][1]]
    tool_payload = json.loads(provider.requests[1][0][-1]["content"])
    assert tool_payload["rows"] == 160 and "path" not in tool_payload
    assert len(storage.list_ml("projects", session.id)) == 1


def test_storage_migration_preserves_sql_sessions(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, dataset_id TEXT, title TEXT, working_summary TEXT, created_at TEXT, updated_at TEXT)")
        db.execute("INSERT INTO sessions VALUES ('old', 'ds', 'Old conversation', '', '2026', '2026')")
    storage = Storage(path)
    storage.initialize()
    storage.initialize()
    assert storage.get_session("old").mode == "sql"


def test_interrupted_runs_recover_without_losing_active_model(settings, storage):
    session, service = setup_ml(settings, storage)
    project = service.project()
    project.update(status="training", active_model_id="previous-model")
    storage.put_ml("projects", project)
    storage.put_ml("runs", {"id": "run-interrupted", "session_id": session.id, "status": "running"})
    storage.recover_ml_runs()
    assert service.project()["status"] == "interrupted"
    assert service.project()["active_model_id"] == "previous-model"
    assert storage.get_ml("runs", "run-interrupted", session.id)["status"] == "interrupted"


@pytest.mark.asyncio
async def test_cancelled_queued_run_is_recorded(settings, storage):
    from app.ml.service import RUNTIME_SLOT
    session, service = setup_ml(settings, storage)
    await call(service, "create_dataset", {})
    await RUNTIME_SLOT.acquire()
    pending = asyncio.create_task(call(service, "train_model", {"task": "classification", "target": "churned"}))
    try:
        await asyncio.sleep(.05)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert storage.list_ml("runs", session.id)[0]["status"] == "cancelled"
    finally:
        RUNTIME_SLOT.release()


@pytest.mark.asyncio
async def test_sql_agent_does_not_offer_or_execute_ml_tools(settings, storage):
    session, _ = setup_ml(settings, storage)
    sql_session = storage.create_session(session.dataset_id, "SQL only", "sql")
    storage.add_message(sql_session.id, "user", "Train something")
    class Provider:
        count = 0
        async def complete(self, messages, tools, role="primary"):
            assert {t["function"]["name"] for t in tools} == {"run_sql", "create_chart"}
            self.count += 1
            if self.count == 1:
                return AssistantTurn(tool_calls=[ToolCall(id="wrong-mode", name="train_model", arguments={"task": "classification", "target": "churned"})])
            assert "only in ML Agent" in messages[-1]["content"]
            return AssistantTurn(content="Training is available in ML Agent mode.")
    events = [e async for e in run_analysis(session_id=sql_session.id, storage=storage, provider=Provider(), settings=settings)]
    assert any(e.type == "tool_finished" and not e.data["ok"] for e in events)
    assert storage.list_ml("runs", sql_session.id) == []


@pytest.mark.asyncio
async def test_api_rejects_cross_mode_session(tmp_path):
    import httpx
    from app.config import Settings
    app = create_app(Settings(data_dir=tmp_path / "data"))
    async with app.router.lifespan_context(app):
        state = app.state.services
        session, _ = setup_ml(state.settings, state.storage)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post("/api/chat/stream", json={"dataset_id": session.dataset_id, "session_id": session.id, "mode": "sql", "message": "Hello"})
            assert response.status_code == 409
            detail = (await client.get(f"/api/sessions/{session.id}")).json()
            assert detail["mode"] == "ml" and detail["project"]["status"] == "new"
