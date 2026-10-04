from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from collections.abc import AsyncIterator
from typing import Any

from .config import Settings
from .context import build_model_messages, compact_conversation
from .models import AgentEvent, AssistantTurn
from .provider import ModelProvider, assistant_message
from .sql_tool import execute_read_only_query
from .storage import Storage
from .ml.contracts import ML_TOOLS, TOOL_MODELS
from .ml.datasets import DatasetService
from .ml.service import MLService, model_summary
from .research import RESEARCH_TOOLS, execute_research_tool

CHART_TYPES = {"kpi", "table", "line", "bar", "area", "scatter", "histogram"}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only DuckDB SELECT query against the table named data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "DuckDB SELECT query"},
                    "purpose": {
                        "type": "string",
                        "description": "Short user-facing activity, e.g. Comparing monthly revenue",
                    },
                },
                "required": ["query", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create a safe chart from a prior SQL result when it improves the analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(CHART_TYPES)},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "series": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["result_id", "type", "x", "title"],
                "additionalProperties": False,
            },
        },
    },
]


async def run_analysis(
    *,
    session_id: str,
    storage: Storage,
    provider: ModelProvider,
    settings: Settings,
) -> AsyncIterator[AgentEvent]:
    session = storage.get_session(session_id)
    ml = MLService(storage, settings, session_id) if session.mode == "ml" else None
    tools = deepcopy(RESEARCH_TOOLS if session.mode == "chat" else TOOLS)
    if ml:
        tools[0]["function"]["parameters"]["properties"]["dataset_id"] = {"type": "string", "description": "Optional ML dataset or prediction reference from this conversation. Its table is also named data."}
        tools.extend(ML_TOOLS)
    dataset = storage.get_dataset(session.dataset_id)
    conversation = storage.list_messages(session_id)
    model_messages = build_model_messages(
        dataset=dataset,
        working_summary=session.working_summary,
        messages=conversation,
        recent_limit=settings.recent_message_limit,
        mode=session.mode,
        ml_context=ml.context() if ml else None,
    )
    yield event(session_id, "analysis_started", message="Looking into it…")

    tool_call_count = 0
    usage: dict[str, int] = {}
    final_text = ""
    final_was_streamed = False
    for iteration in range(settings.max_analysis_iterations):
        turn: AssistantTurn | None = None
        streamed_this_turn = False
        stream_method = getattr(provider, "stream_complete", None)
        if callable(stream_method):
            async for part in stream_method(model_messages, tools, "primary"):
                if isinstance(part, AssistantTurn):
                    turn = part
                else:
                    streamed_this_turn = True
                    yield event(session_id, "message_delta", delta=part)
        else:
            turn = await provider.complete(model_messages, tools, "primary")
        if turn is None:
            raise RuntimeError("Model stream ended without a completed turn")
        merge_usage(usage, turn)
        model_messages.append(assistant_message(turn))
        if not turn.tool_calls:
            final_text = (turn.content or "").strip()
            final_was_streamed = streamed_this_turn
            break

        for call in turn.tool_calls:
            tool_call_count += 1
            if tool_call_count > settings.max_tool_calls:
                final_text = "I reached the configured tool-call limit before completing the analysis."
                break
            purpose = clean_activity(call.arguments.get("purpose") or default_activity(call.name))
            yield event(
                session_id,
                "tool_started",
                tool_call_id=call.id,
                tool=call.name,
                label=purpose,
            )
            try:
                if call.name in {"web_search", "read_web_page"}:
                    if session.mode != "chat":
                        raise ValueError("Research tools are available only in Research Chat mode")
                    tool_payload = await execute_research_tool(call.name, call.arguments, settings)
                    artifact_event = None
                elif call.name in TOOL_MODELS:
                    if ml is None:
                        raise ValueError("ML tools are available only in ML Agent mode")
                    artifact_event = None
                    tool_payload = {}
                    async for update in ml.execute(call.name, call.arguments):
                        if update["type"] == "progress":
                            yield event(session_id, "ml_progress", tool_call_id=call.id, **update)
                        else:
                            result = update["result"]
                            artifact = storage.add_artifact(session_id, "ml_result", result)
                            artifact_event = event(session_id, "ml_result", **result, artifact_id=artifact.id)
                            tool_payload = model_summary(result)
                else:
                    tool_payload, artifact_event = await execute_tool(
                        call.name, call.arguments, session_id=session_id,
                        duckdb_path=dataset.duckdb_path, storage=storage, settings=settings,
                    )
                yield event(
                    session_id,
                    "tool_finished",
                    tool_call_id=call.id,
                    tool=call.name,
                    label=purpose,
                    ok=True,
                    duration_ms=tool_payload.get("duration_ms"),
                    rows_returned=tool_payload.get("returned_row_count"),
                    sql=call.arguments.get("query"),
                )
                if artifact_event:
                    yield artifact_event
                tool_content = {"ok": True, **tool_payload}
            except Exception as exc:
                message = safe_error(exc)
                tool_content = {"ok": False, "error": message}
                yield event(
                    session_id,
                    "tool_finished",
                    tool_call_id=call.id,
                    tool=call.name,
                    label=purpose,
                    ok=False,
                    error=message,
                    sql=call.arguments.get("query"),
                )
            model_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(tool_content, default=str, separators=(",", ":")),
                }
            )
        if final_text:
            break
    else:
        final_messages = model_messages + [
            {
                "role": "system",
                "content": "Tool iterations are exhausted. Give the best concise answer from results so far.",
            }
        ]
        final_turn: AssistantTurn | None = None
        stream_method = getattr(provider, "stream_complete", None)
        if callable(stream_method):
            async for part in stream_method(final_messages, [], "primary"):
                if isinstance(part, AssistantTurn):
                    final_turn = part
                else:
                    final_was_streamed = True
                    yield event(session_id, "message_delta", delta=part)
        else:
            final_turn = await provider.complete(final_messages, [], "primary")
        if final_turn is None:
            raise RuntimeError("Model stream ended without a completed turn")
        merge_usage(usage, final_turn)
        final_text = (final_turn.content or "").strip()

    if not final_text:
        final_text = "I could not produce a grounded answer from the available results."
    storage.add_message(session_id, "assistant", final_text)
    all_messages = storage.list_messages(session_id)
    if sum(len(item.content) for item in all_messages) > settings.compaction_threshold_chars:
        storage.update_summary(
            session_id,
            compact_conversation(
                all_messages,
                recent_limit=settings.recent_message_limit,
                max_chars=settings.max_summary_chars,
            ),
        )
    if not final_was_streamed:
        for chunk in text_chunks(final_text):
            yield event(session_id, "message_delta", delta=chunk)
            await asyncio.sleep(0)
    yield event(
        session_id,
        "analysis_complete",
        message=final_text,
        tool_calls=tool_call_count,
        usage=usage,
    )


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    session_id: str,
    duckdb_path: str,
    storage: Storage,
    settings: Settings,
) -> tuple[dict[str, Any], AgentEvent | None]:
    if name == "run_sql":
        if arguments.get("dataset_id"):
            if storage.get_session(session_id).mode != "ml":
                raise ValueError("ML dataset references require ML Agent mode")
            duckdb_path = DatasetService(storage, settings, session_id).get(arguments["dataset_id"])["duckdb_path"]
        query = arguments.get("query")
        if not isinstance(query, str):
            raise ValueError("run_sql requires a query string")
        # DuckDB enforces its own interrupt timer. Offload the bounded query
        # so chat cancellation and other requests remain responsive.
        result = await asyncio.to_thread(execute_read_only_query,
            duckdb_path,
            query,
            row_limit=settings.query_row_limit,
            timeout_seconds=settings.query_timeout_seconds,
        )
        payload = {"sql": query, **result.as_payload()}
        artifact = storage.add_artifact(session_id, "query_result", payload)
        payload["result_id"] = artifact.id
        return payload, event(session_id, "query_result", **payload)

    if name == "create_chart":
        result_id = arguments.get("result_id")
        source = storage.get_artifact(result_id)
        if source.session_id != session_id or source.kind != "query_result":
            raise ValueError("Charts must reference a query result from this analysis")
        chart_type = arguments.get("type")
        if chart_type not in CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        columns = source.payload.get("columns", [])
        x = arguments.get("x")
        y = arguments.get("y")
        series = arguments.get("series")
        if not x or (chart_type not in {"table", "kpi"} and not y):
            raise ValueError("Provide an x column and a numeric y column for this chart")
        if y and chart_type != "table":
            index = columns.index(y) if y in columns else -1
            if index < 0 or not any(isinstance(row[index], (int, float)) and not isinstance(row[index], bool) for row in source.payload.get("rows", [])):
                raise ValueError("The y column must contain numeric values")
        for field in [x, y, series]:
            if field and field not in columns:
                raise ValueError(f"Chart field {field!r} is not in result {result_id}")
        spec = {
            "result_id": result_id,
            "type": chart_type,
            "x": x,
            "y": y,
            "series": series,
            "title": str(arguments.get("title") or "Analysis chart")[:160],
        }
        artifact = storage.add_artifact(session_id, "chart", spec)
        payload = {"chart_id": artifact.id, "spec": spec, "source": source.payload}
        return payload, event(session_id, "chart_created", **payload)

    raise ValueError(f"Unknown tool: {name}")


def event(session_id: str, event_type: str, **data: Any) -> AgentEvent:
    return AgentEvent(type=event_type, session_id=session_id, data=data)


def merge_usage(total: dict[str, int], turn: AssistantTurn) -> None:
    for key, value in turn.usage.items():
        total[key] = total.get(key, 0) + value


def clean_activity(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:120] or "Running analysis"


def default_activity(tool_name: str) -> str:
    return {"web_search": "Searching the web", "read_web_page": "Reading a source", "create_chart": "Creating visualization", "run_sql": "Querying the dataset", "create_dataset": "Preparing dataset", "inspect_dataset": "Checking data quality", "train_model": "Building and evaluating models", "predict": "Running predictions", "cluster_dataset": "Finding customer segments", "forecast_series": "Forecasting future values", "detect_anomalies": "Checking for unusual patterns"}.get(tool_name, tool_name.replace("_", " ").capitalize())


def safe_error(exc: Exception) -> str:
    if not isinstance(exc, (ValueError, TimeoutError, KeyError)):
        return "The operation could not complete. Check the dataset and query, then try again."
    return re.sub(r"(?:/[\w.\-]+){2,}", "[local path]", str(exc).split("Stack trace")[0])[:600]


def text_chunks(text: str, size: int = 80) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]
