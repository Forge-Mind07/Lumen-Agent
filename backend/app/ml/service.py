from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..models import new_id, utc_now
from ..storage import Storage
from .contracts import TOOL_MODELS, TrainModel
from .datasets import DatasetService, public_record

RUNTIME_SLOT = asyncio.Semaphore(1)
TRAINING_TOOLS = {"train_model", "cluster_dataset", "forecast_series", "detect_anomalies"}


class MLService:
    def __init__(self, storage: Storage, settings: Settings, session_id: str):
        session = storage.get_session(session_id)
        if session.mode != "ml":
            raise ValueError("ML tools are available only in ML Agent mode")
        self.storage, self.settings, self.session_id = storage, settings, session_id
        self.datasets = DatasetService(storage, settings, session_id)

    def project(self):
        projects = self.storage.list_ml("projects", self.session_id)
        if projects:
            return projects[0]
        session = self.storage.get_session(self.session_id)
        return self.storage.put_ml("projects", {"id": new_id("proj"), "session_id": self.session_id,
            "name": session.title, "status": "new", "dataset_id": None, "target": None,
            "feature_columns": [], "excluded_columns": [], "active_model_id": None,
            "latest_run_id": None, "created_at": utc_now()})

    def context(self):
        project = self.project()
        return {"project": project, "recent_models": [public_record(m) for m in self.storage.list_ml("models", self.session_id)[:4]],
                "datasets": [public_record(d) for d in self.storage.list_ml("datasets", self.session_id)[:6]]}

    async def execute(self, name: str, raw: dict[str, Any]):
        if name not in TOOL_MODELS:
            raise ValueError("Unknown ML tool")
        cls = TOOL_MODELS[name][0]
        try:
            supplied = cls.model_validate(raw)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise ValueError("Invalid tool arguments: " + errors[:600]) from None
        args = supplied.model_dump()
        project = self.project()
        if name == "create_dataset":
            yield {"type": "progress", "stage": "preparing", "label": "Materializing dataset", "status": "running"}
            record = await asyncio.to_thread(self.datasets.materialize, args["query"], args["name"])
            project.update(dataset_id=record["id"], status="dataset_ready")
            self.storage.put_ml("projects", project)
            yield {"type": "result", "result": {**public_record(record), "summary": f"Prepared {record['rows']:,} rows for modeling."}}
            return
        if name in {"compare_models", "list_project_models"}:
            models = self.storage.list_ml("models", self.session_id)
            if args.get("model_ids"):
                models = [self.storage.get_ml("models", mid, self.session_id) for mid in args["model_ids"]]
            signatures = {(m.get("dataset_fingerprint"), m.get("target"), json.dumps(m.get("split"), sort_keys=True), m.get("random_seed")) for m in models}
            yield {"type": "result", "result": {"models": [public_record(m) for m in models[:10]], "active_model_id": project.get("active_model_id"),
                "summary": f"{len(models)} saved models in this project.",
                "warnings": ["Different datasets, targets or splits are not directly comparable."] if len(signatures) > 1 else []}}
            return
        model = None
        if name in {"predict", "evaluate_model", "explain_model", "explain_prediction"}:
            model_id = args.get("model_id") or project.get("active_model_id")
            if not model_id:
                raise ValueError("Train a model before this operation.")
            model = self.storage.get_ml("models", model_id, self.session_id)
            if name == "explain_model" or (name == "evaluate_model" and not args.get("dataset_id")):
                run = self.storage.get_ml("runs", model["run_id"], self.session_id)
                yield {"type": "result", "result": {**run["result"], "summary": "Saved model evaluation and measured feature importance."}}
                return
        if name == "train_model":
            previous = project.get("training_config", {})
            overrides = supplied.model_dump(exclude_unset=True)
            args = TrainModel.model_validate({**previous, **overrides}).model_dump()
            if not args.get("task") or not args.get("target"):
                raise ValueError("Specify the task and target column; inspect the dataset if their meaning is unclear.")
            if len(args.get("algorithms") or ["default", "default"]) + 1 > self.settings.ml_max_candidates:
                raise ValueError("Candidate limit exceeded, including the required baseline.")
        dataset_id = args.get("dataset_id") or project.get("dataset_id")
        dataset = self.datasets.get(dataset_id) if dataset_id else None
        if dataset is None and args.get("row") is None:
            raise ValueError("Create an ML dataset with create_dataset before this operation.")
        if dataset is not None:
            args["dataset_id"] = dataset["id"]
        run_id = new_id("run")
        directory = (self.settings.data_dir / "ml" / "runs" / run_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        messages = self.storage.list_messages(self.session_id)
        run = {"id": run_id, "run_id": run_id, "session_id": self.session_id, "project_id": project["id"],
               "operation": name, "status": "queued", "created_at": utc_now(),
               "user_request": next((m.content for m in reversed(messages) if m.role == "user"), "API operation"),
               "dataset_id": dataset_id, "dataset_query_id": dataset.get("query_id") if dataset else None,
               "configuration": args, "dataset_fingerprint": dataset.get("fingerprint") if dataset else None}
        self.storage.put_ml("runs", run)
        job_path = directory / "job.json"
        job_path.write_text(json.dumps({"operation": name, "arguments": args, "dataset": dataset, "model": model}))
        yield {"type": "progress", "run_id": run_id, "stage": "queued", "label": "Waiting for ML runtime", "status": "running"}
        process = None
        try:
            async with RUNTIME_SLOT:
                run.update(status="running")
                self.storage.put_ml("runs", run)
                if name in TRAINING_TOOLS:
                    project.update(status="training", latest_run_id=run_id)
                    self.storage.put_ml("projects", project)
                env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                async with asyncio.timeout(self.settings.ml_timeout_seconds):
                    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "app.ml.worker", str(job_path),
                        cwd=str(Path(__file__).resolve().parents[2]), env=env,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                    async for line in process.stdout:
                        progress = json.loads(line)
                        yield {"type": "progress", "run_id": run_id, **progress}
                    await process.wait()
                    result_file = directory / "result.json"
                    if process.returncode or not result_file.exists():
                        raise ValueError("ML worker stopped unexpectedly. Try a smaller dataset or simpler model.")
                    response = json.loads(result_file.read_text())
                    if not response["ok"]:
                        raise ValueError(response["error"])
                    result = response["result"]
                result.update(run_id=run_id, project_id=project["id"], status="completed")
                if dataset:
                    result["dataset"] = public_record(dataset)
                if name in TRAINING_TOOLS:
                    model_id = new_id("model")
                    model_record = {"id": model_id, "model_id": model_id, "session_id": self.session_id, "project_id": project["id"],
                        "run_id": run_id, "name": f"{result['algorithm'].replace('_', ' ')} · {run['created_at'][:10]}",
                        "status": "ready", "created_at": utc_now(), "dataset_fingerprint": dataset["fingerprint"],
                        "random_seed": args.get("random_seed"),
                        **{k: result[k] for k in ("algorithm", "task", "target", "features", "feature_schema", "metrics", "split", "preprocessing", "artifact_path", "positive_class") if k in result}}
                    self.storage.put_ml("models", model_record)
                    result["model_id"] = model_id
                    project.update(active_model_id=model_id, latest_run_id=run_id, status="trained", task=result["task"],
                                   target=result.get("target"), feature_columns=result.get("features", []),
                                   excluded_columns=list(result.get("excluded_features", {})))
                    if name == "train_model":
                        project["training_config"] = args
                    self.storage.put_ml("projects", project)
                for filename, ref_key in [("predictions.json", "prediction_dataset_id"), ("evaluation.json", "evaluation_dataset_id")]:
                    file = directory / filename
                    if file.exists():
                        payload = json.loads(file.read_text())
                        rows, columns = payload["data"], payload["columns"]
                        if rows:
                            types = ["DOUBLE" if all(r[i] is None or isinstance(r[i], (float, int)) for r in rows[:100]) else "VARCHAR" for i in range(len(columns))]
                            derived = await asyncio.to_thread(self.datasets.save, columns, types, rows, name=filename.removesuffix(".json"), source_dataset_id=dataset_id, source_model_id=result.get("model_id"), source_run_id=run_id)
                            result[ref_key] = derived["id"]
                        if filename == "predictions.json":
                            preview = {"columns": columns, "column_types": types if rows else ["VARCHAR"]*len(columns), "rows": rows[:args.get("limit", 100)],
                                       "returned_row_count": min(len(rows), args.get("limit", 100)), "truncated": len(rows) > args.get("limit", 100), "duration_ms": 0, "sql": ""}
                            artifact = self.storage.add_artifact(self.session_id, "query_result", preview)
                            preview["result_id"] = artifact.id
                            result["predictions"] = preview
                result.pop("artifact_path", None)
                run.update(status="completed", completed_at=utc_now(), result=result)
                self.storage.put_ml("runs", run)
                yield {"type": "progress", "run_id": run_id, "stage": "completed", "label": "Complete", "status": "completed"}
                yield {"type": "result", "result": result}
        except BaseException as exc:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            cancelled = isinstance(exc, (asyncio.CancelledError, GeneratorExit))
            message = "Operation cancelled" if cancelled else "ML time limit exceeded. Try fewer rows or a simpler model." if isinstance(exc, TimeoutError) else str(exc)[:600]
            run.update(status="cancelled" if cancelled else "failed", error=message, completed_at=utc_now())
            self.storage.put_ml("runs", run)
            if name in TRAINING_TOOLS:
                project["status"] = run["status"]
                self.storage.put_ml("projects", project)
            if cancelled:
                raise
            raise ValueError(message) from None


def model_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep numerical evidence and provenance; bound row/point payloads to the LLM."""
    compact = {k: v for k, v in result.items() if k not in {"charts", "predictions", "models"}}
    if "feature_importance" in compact:
        compact["feature_importance"] = compact["feature_importance"][:15]
    if "predictions" in result:
        table = result["predictions"]
        compact["predictions"] = {**table, "rows": table["rows"][:3]}
    if "models" in result:
        compact["models"] = result["models"][:10]
    return compact
