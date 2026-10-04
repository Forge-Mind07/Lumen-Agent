from __future__ import annotations

import csv
import io
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent import run_analysis, safe_error
from .config import Settings
from .datasets import register_dataset, remove_dataset_files
from .models import ChatRequest, DatasetRecord
from .provider import ModelProvider, OpenAICompatibleProvider
from .storage import CHAT_DATASET_ID, Storage
from .ml.service import MLService
from .ml.datasets import public_record


@dataclass(slots=True)
class AppServices:
    settings: Settings
    storage: Storage
    provider: ModelProvider
    active_sessions: set[str] = field(default_factory=set)


def create_app(
    settings: Settings | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.data_dir.mkdir(parents=True, exist_ok=True)
        storage = Storage(config.sqlite_path)
        storage.initialize()
        storage.recover_ml_runs()
        app.state.services = AppServices(
            settings=config,
            storage=storage,
            provider=provider or OpenAICompatibleProvider(config),
        )
        yield

    app = FastAPI(title="Local Analyst", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[item.strip() for item in config.cors_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/datasets", response_model=list[DatasetRecord])
    async def list_datasets(request: Request) -> list[DatasetRecord]:
        return services(request).storage.list_datasets()

    @app.post("/api/datasets", response_model=DatasetRecord)
    async def upload_dataset(request: Request, file: UploadFile = File(...)) -> DatasetRecord:
        if not file.filename:
            raise HTTPException(400, "The uploaded file needs a filename")
        state = services(request)
        try:
            return register_dataset(
                filename=file.filename,
                source=file.file,
                storage=state.storage,
                settings=state.settings,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetRecord)
    async def get_dataset(dataset_id: str, request: Request) -> DatasetRecord:
        try:
            return services(request).storage.get_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    async def delete_dataset(dataset_id: str, request: Request) -> Response:
        state = services(request)
        if dataset_id == CHAT_DATASET_ID:
            raise HTTPException(400, "The Research Chat workspace cannot be deleted")
        active = set(state.storage.dataset_session_ids(dataset_id)) & state.active_sessions
        if active:
            raise HTTPException(409, "Stop the active conversation before deleting this dataset")
        try:
            _, paths = state.storage.delete_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        remove_dataset_files(paths, state.settings.data_dir)
        return Response(status_code=204)

    @app.get("/api/sessions")
    async def list_sessions(request: Request) -> list[dict[str, Any]]:
        return [item.model_dump() for item in services(request).storage.list_sessions()]

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, request: Request) -> dict[str, Any]:
        storage = services(request).storage
        try:
            session = storage.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            **session.model_dump(),
            "messages": [item.model_dump() for item in storage.list_messages(session_id)],
            "artifacts": [item.model_dump() for item in storage.list_artifacts(session_id)],
            "project": MLService(storage, services(request).settings, session_id).project() if session.mode == "ml" else None,
        }

    @app.get("/api/ml/projects/{session_id}")
    async def ml_project(session_id: str, request: Request):
        state = services(request)
        try:
            service = MLService(state.storage, state.settings, session_id)
            return {**service.context(), "runs": [public_record(r) for r in state.storage.list_ml("runs", session_id)]}
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, safe_error(exc)) from exc

    @app.get("/api/ml/runs/{run_id}")
    async def ml_run(run_id: str, session_id: str, request: Request):
        try:
            return public_record(services(request).storage.get_ml("runs", run_id, session_id))
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/chat/stream")
    async def chat_stream(chat: ChatRequest, request: Request) -> StreamingResponse:
        state = services(request)
        if chat.session_id:
            try:
                session = state.storage.get_session(chat.session_id)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            requested_dataset_id = CHAT_DATASET_ID if chat.mode == "chat" else chat.dataset_id
            if session.dataset_id != requested_dataset_id:
                raise HTTPException(409, "This session belongs to a different dataset")
            if session.mode != chat.mode:
                raise HTTPException(409, "Start a separate conversation when switching agent modes")
        else:
            requested_dataset_id = CHAT_DATASET_ID if chat.mode == "chat" else chat.dataset_id
            if not requested_dataset_id:
                raise HTTPException(422, "Choose a dataset for SQL or ML mode")
            session = state.storage.create_session(requested_dataset_id, chat.message, chat.mode)
        try:
            dataset = state.storage.get_dataset(session.dataset_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if dataset.status != "ready":
            raise HTTPException(409, "The dataset is not ready for analysis")
        if not chat.message.strip():
            raise HTTPException(422, "Please enter a message")
        if session.id in state.active_sessions:
            raise HTTPException(409, "This conversation already has an active request")
        state.active_sessions.add(session.id)
        state.storage.add_message(session.id, "user", chat.message.strip())

        async def stream():
            initial = {
                "version": 1,
                "id": f"evt_session_{session.id}",
                "type": "session_started",
                "timestamp": session.created_at,
                "session_id": session.id,
                "data": {"session": session.model_dump()},
            }
            yield sse("session_started", initial)
            try:
                async for agent_event in run_analysis(
                    session_id=session.id,
                    storage=state.storage,
                    provider=state.provider,
                    settings=state.settings,
                ):
                    if await request.is_disconnected():
                        break
                    yield sse(agent_event.type, agent_event.model_dump())
            except Exception as exc:
                error_event = {
                    "version": 1,
                    "id": f"evt_error_{session.id}",
                    "type": "error",
                    "session_id": session.id,
                    "data": {"message": safe_error(exc)},
                }
                yield sse("error", error_event)
            finally:
                state.active_sessions.discard(session.id)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/artifacts/{artifact_id}/csv")
    async def download_artifact(artifact_id: str, request: Request) -> Response:
        try:
            artifact = services(request).storage.get_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if artifact.kind != "query_result":
            raise HTTPException(400, "Only query result artifacts can be downloaded as CSV")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(artifact.payload.get("columns", []))
        writer.writerows(artifact.payload.get("rows", []))
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{artifact_id}.csv"'},
        )

    return app


def services(request: Request) -> AppServices:
    return request.app.state.services


def sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"


app = create_app()
