from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

import httpx

from .config import Settings
from .models import AssistantTurn, ToolCall


class ModelProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: Literal["primary", "secondary"] = "primary",
    ) -> AssistantTurn: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: Literal["primary", "secondary"] = "primary",
    ) -> AssistantTurn:
        endpoint = self.settings.model_endpoint(role)
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=self.settings.model_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:1000]
            raise RuntimeError(f"Model server returned {response.status_code}: {detail}") from exc
        body = response.json()
        if not body.get("choices"):
            raise RuntimeError("Model server returned no choices")
        message = body["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = (
                    raw_arguments if isinstance(raw_arguments, dict) else json.loads(raw_arguments)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Model returned invalid arguments for {function.get('name', 'a tool')}"
                ) from exc
            tool_calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{len(tool_calls) + 1}",
                    name=function.get("name") or "",
                    arguments=arguments,
                )
            )
        usage = body.get("usage") or {}
        return AssistantTurn(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
            model=body.get("model") or endpoint.model,
        )

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: Literal["primary", "secondary"] = "primary",
    ) -> AsyncIterator[str | AssistantTurn]:
        """Yield visible answer text as it arrives, followed by the parsed turn.

        Reasoning-specific fields exposed by some local servers are deliberately
        ignored. Tool calls are assembled silently and returned in the final turn.
        """
        endpoint = self.settings.model_endpoint(role)
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        content_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        usage: dict[str, int] = {}
        response_model = endpoint.model

        stream_timeout = httpx.Timeout(
            self.settings.model_timeout_seconds,
            read=self.settings.stream_idle_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(errors="replace")[:1000]
                    raise RuntimeError(f"Model server returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    body = json.loads(data)
                    response_model = body.get("model") or response_model
                    for key, value in (body.get("usage") or {}).items():
                        if isinstance(value, int):
                            usage[key] = value
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    visible = delta.get("content")
                    if isinstance(visible, str) and visible:
                        content_parts.append(visible)
                        yield visible
                    for fragment in delta.get("tool_calls") or []:
                        index = int(fragment.get("index", 0))
                        part = tool_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if fragment.get("id"):
                            part["id"] += fragment["id"]
                        function = fragment.get("function") or {}
                        if function.get("name"):
                            part["name"] += function["name"]
                        if function.get("arguments"):
                            part["arguments"] += function["arguments"]

        calls: list[ToolCall] = []
        for index, part in sorted(tool_parts.items()):
            try:
                arguments = json.loads(part["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Model returned invalid arguments for {part['name'] or 'a tool'}") from exc
            calls.append(
                ToolCall(
                    id=part["id"] or f"call_{index + 1}",
                    name=part["name"],
                    arguments=arguments,
                )
            )
        yield AssistantTurn(
            content="".join(content_parts) or None,
            tool_calls=calls,
            usage=usage,
            model=response_model,
        )


def assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in turn.tool_calls
        ]
    return message
