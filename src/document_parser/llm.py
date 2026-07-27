from __future__ import annotations

from contextlib import contextmanager
import json
import os
import signal
import threading
from typing import Any

import requests

from .config import RuntimeConfig


DEFAULT_LLM_MAX_TOKENS = 8192


class LlmError(RuntimeError):
    pass


class LlmClient:
    def structured_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def structured_json_with_max_tokens(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        del max_tokens
        return self.structured_json(system, user, schema)


class OpenAICompatibleLlmClient(LlmClient):
    def __init__(self, config: RuntimeConfig, timeout_seconds: int = 180) -> None:
        self._api_key = config.llm_api_key
        self._base_url = config.llm_base_url.rstrip("/")
        self._model = config.llm_model
        self._timeout_seconds = timeout_seconds

    def structured_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self._structured_json(system, user, schema, DEFAULT_LLM_MAX_TOKENS)

    def structured_json_with_max_tokens(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        return self._structured_json(system, user, schema, max_tokens)

    def _structured_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        if not self._api_key or not self._base_url or not self._model:
            raise LlmError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL are required for online LLM agent mode.")
        payload = _chat_payload(self._model, system, user, max_tokens)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "span_grounded_agent_items",
                "schema": schema,
                "strict": True,
            },
        }
        response = self._post_with_timeout_retry(payload)
        if response.status_code == 400:
            fallback_payload = _chat_payload(self._model, system, user, max_tokens)
            fallback_payload["response_format"] = {"type": "json_object"}
            response = self._post_with_timeout_retry(fallback_payload)
        if response.status_code == 400:
            plain_payload = _chat_payload(self._model, system, user, max_tokens)
            response = self._post_with_timeout_retry(plain_payload)
        if response.status_code != 200:
            raise LlmError(f"LLM request failed with HTTP {response.status_code}")
        body = response.json()
        try:
            return _structured_json_from_response(body)
        except LlmError as exc:
            if "message content is empty" in str(exc):
                raise
            if "not valid JSON" not in str(exc):
                raise
            malformed_content = _message_content(body)
            if "{" not in malformed_content:
                raise
            retry_payload = _chat_payload(
                self._model,
                system + " Return one complete valid JSON object only. Keep the response compact and do not use markdown.",
                user,
                max_tokens,
            )
            retry_payload["response_format"] = {"type": "json_object"}
            retry_response = self._post_with_timeout_retry(retry_payload)
            if retry_response.status_code != 200:
                raise LlmError(f"LLM JSON retry failed with HTTP {retry_response.status_code}") from exc
            return _structured_json_from_response(retry_response.json())

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        try:
            with _request_deadline(self._timeout_seconds):
                return requests.post(
                    _chat_completions_url(self._base_url),
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout_seconds,
                )
        except requests.RequestException as exc:
            raise LlmError(f"LLM request failed: {exc.__class__.__name__}") from exc

    def _post_with_timeout_retry(self, payload: dict[str, Any]) -> requests.Response:
        try:
            return self._post(payload)
        except LlmError as exc:
            if "exceeded" not in str(exc) and "Timeout" not in str(exc):
                raise
            return self._post(payload)


def _chat_payload(model: str, system: str, user: str, default_max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": _llm_max_tokens(default_max_tokens),
    }


def _llm_max_tokens(default: int = DEFAULT_LLM_MAX_TOKENS) -> int:
    raw_value = os.getenv("LLM_MAX_TOKENS", str(default))
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


@contextmanager
def _request_deadline(timeout_seconds: int):
    if threading.current_thread() is not threading.main_thread() or timeout_seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):  # type: ignore[no-untyped-def]
        del signum, frame
        raise LlmError(f"LLM request exceeded {timeout_seconds} seconds.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    version_segment = base_url.rsplit("/", 1)[-1]
    if len(version_segment) > 1 and version_segment[0] == "v" and version_segment[1:].isdigit():
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _structured_json_from_response(body: dict[str, Any]) -> dict[str, Any]:
    content = _message_content(body)
    if not content:
        parsed = _message_parsed(body)
        if isinstance(parsed, dict):
            return parsed
        raise LlmError("LLM response message content is empty.")
    content = _json_object_text(content)
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LlmError("LLM response content was not valid JSON.") from exc
    if not isinstance(parsed_content, dict):
        raise LlmError("LLM response JSON root must be an object.")
    return parsed_content


def _message_parsed(body: dict[str, Any]) -> Any:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmError("LLM response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LlmError("LLM response did not include choices[0].message.")
    return message.get("parsed")


def _message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmError("LLM response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LlmError("LLM response did not include choices[0].message.")
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def _json_object_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped
