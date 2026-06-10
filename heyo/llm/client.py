"""Async OpenAI-compatible chat client.

Works against any OpenAI-compatible server: Ollama (/v1) and vLLM. Which backend
serves which role is decided by models.yaml, so swapping or mixing backends is
pure configuration.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from heyo.config import ModelsConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, models: ModelsConfig, timeout: float = 120.0):
        self.models = models
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
        think: bool = True,
    ) -> dict[str, Any]:
        """One chat completion. Returns the assistant message dict
        (keys: content, optionally tool_calls).

        think=False prepends Qwen's /no_think soft switch — reasoning models burn
        seconds of hidden tokens otherwise; harmless no-op for other models."""
        if not think:
            messages = [{"role": "system", "content": "/no_think"}, *messages]
        payload: dict[str, Any] = {
            "model": self.models.role(role).model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        url = f"{self.models.base_url(role)}/chat/completions"
        resp = await self._http.post(url, json=payload)
        if resp.status_code != 200:
            raise LLMError(f"{url} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]

    async def chat_structured(
        self,
        role: str,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        temperature: float = 0.0,
        think: bool = True,
    ) -> dict[str, Any]:
        """Chat completion parsed as JSON matching json_schema.

        Prompt-based JSON is the primary path: Ollama's grammar-constrained
        response_format is an order of magnitude slower on consumer GPUs
        (measured 42s vs 4s on a GTX 1660 Ti). Grammar mode is the fallback
        when the model's freeform JSON doesn't parse.
        """
        prompted = messages + [
            {
                "role": "system",
                "content": "Respond ONLY with a JSON object matching this schema, no prose: "
                + json.dumps(json_schema),
            }
        ]
        try:
            msg = await self.chat(role, prompted, temperature=temperature, think=think)
            return _parse_json_content(msg.get("content") or "")
        except (LLMError, json.JSONDecodeError):
            msg = await self.chat(
                role, messages, temperature=temperature, json_schema=json_schema, think=think
            )
            return _parse_json_content(msg.get("content") or "")

    async def stream_message(
        self,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.4,
        think: bool = True,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Streaming chat completion with live reasoning and tool-call assembly.

        Yields ("thinking", text) for <think> block tokens, ("token", text) for
        answer tokens, then exactly one ("message", assistant_message) with the
        think-stripped content and any accumulated tool_calls.
        """
        if not think:
            messages = [{"role": "system", "content": "/no_think"}, *messages]
        payload: dict[str, Any] = {
            "model": self.models.role(role).model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        url = f"{self.models.base_url(role)}/chat/completions"

        parser = _ThinkParser()
        content_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}

        async with self._http.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise LLMError(f"{url} -> {resp.status_code}: {body[:500]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    cur = calls.setdefault(
                        idx, {"id": tc.get("id") or f"call_{idx}",
                              "function": {"name": "", "arguments": ""}}
                    )
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        cur["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        cur["function"]["arguments"] += fn["arguments"]
                # some servers expose reasoning as its own delta field
                if delta.get("reasoning"):
                    yield ("thinking", delta["reasoning"])
                if delta.get("content"):
                    for kind, text in parser.feed(delta["content"]):
                        if kind == "token":
                            content_parts.append(text)
                        yield (kind, text)
        for kind, text in parser.flush():
            if kind == "token":
                content_parts.append(text)
            yield (kind, text)

        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts).strip()}
        if calls:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": c["function"]}
                for _, c in sorted(calls.items())
            ]
        yield ("message", message)

    async def stream(
        self,
        role: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they arrive."""
        payload = {
            "model": self.models.role(role).model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self.models.base_url(role)}/chat/completions"
        async with self._http.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise LLMError(f"{url} -> {resp.status_code}: {body[:500]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]


class _ThinkParser:
    """Incremental splitter of a token stream into thinking vs answer text.

    Reasoning models open with a <think>...</think> block; tags can be split
    across stream chunks, so a small holdback buffer is kept while inside one.
    """

    def __init__(self) -> None:
        self.buf = ""
        self.mode = "detect"  # detect -> think|token

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        out: list[tuple[str, str]] = []
        while True:
            if self.mode == "detect":
                stripped = self.buf.lstrip()
                if not stripped:
                    return out
                if stripped.startswith("<think>"):
                    self.buf = stripped[len("<think>"):]
                    self.mode = "think"
                    continue
                if "<think>".startswith(stripped):
                    return out  # could still become a tag; wait for more
                self.mode = "token"
                continue
            if self.mode == "think":
                end = self.buf.find("</think>")
                if end != -1:
                    if self.buf[:end]:
                        out.append(("thinking", self.buf[:end]))
                    self.buf = self.buf[end + len("</think>"):].lstrip()
                    self.mode = "token"
                    continue
                holdback = len("</think>") - 1
                if len(self.buf) > holdback:
                    out.append(("thinking", self.buf[:-holdback]))
                    self.buf = self.buf[-holdback:]
                return out
            # token mode: pass everything through
            if self.buf:
                out.append(("token", self.buf))
                self.buf = ""
            return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.buf.strip():
            out.append(("thinking" if self.mode == "think" else "token", self.buf))
        self.buf = ""
        return out


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    # Reasoning models may emit <think>...</think> before the JSON.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"model did not return JSON: {content[:200]!r}")
    return json.loads(text[start : end + 1])
