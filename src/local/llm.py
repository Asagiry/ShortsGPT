"""OpenAI-compatible LLM backend used for transcript planning and selection."""
import json
import os
import socket
import time
import urllib.error
import urllib.request

from ..config import llm_base_url, llm_beat_model, llm_fast_model, llm_model, llm_strong_model, require_openai_key
from .progress import user_log


def _content_from_response(body: bytes, content_type: str) -> str:
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type or text.lstrip().startswith("data:"):
        parts = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content") or ""
            if content:
                parts.append(content)
        return "".join(parts)

    data = json.loads(text)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM response has unexpected format: {data}") from e


def _llm_timeout_seconds() -> int:
    try:
        return max(60, min(int(os.getenv("LLM_TIMEOUT_SECONDS", "600")), 3600))
    except ValueError:
        return 600


def _llm_retry_count() -> int:
    try:
        return max(1, min(int(os.getenv("LLM_RETRIES", "3")), 8))
    except ValueError:
        return 3


def _call_llm_once(prompt: str, model_name: str, timeout: int) -> str:
    api_key = require_openai_key()
    base_url = (llm_base_url() or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = json.dumps(
        {
            "model": model_name,
            "temperature": 0.45,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            body = response.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {e.code}: {body[:1000]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM request failed: {e.reason}") from e

    try:
        return _content_from_response(body, content_type)
    except json.JSONDecodeError as e:
        preview = body.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"LLM response is not valid JSON/SSE: {preview}") from e


def _is_retryable_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "remote end closed",
            "http 408",
            "http 409",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def _call_llm(prompt: str, model_name: str) -> str:
    timeout = _llm_timeout_seconds()
    retries = _llm_retry_count()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _call_llm_once(prompt, model_name, timeout)
        except (RuntimeError, urllib.error.URLError, socket.timeout, TimeoutError) as e:
            last_error = e
            if attempt >= retries or not _is_retryable_error(e):
                break
            delay = min(8 * attempt, 30)
            user_log(
                "LLM retry",
                f"{model_name}: attempt {attempt}/{retries} failed ({e}); retrying in {delay}s",
            )
            time.sleep(delay)
    raise RuntimeError(f"LLM request failed after {retries} attempt(s): {last_error}") from last_error


def call_openai_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_model())


def call_fast_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_fast_model())


def call_beat_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_beat_model())


def call_strong_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_strong_model())
