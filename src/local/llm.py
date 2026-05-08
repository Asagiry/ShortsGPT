"""OpenAI-compatible LLM backend used for transcript planning and selection."""
import json
import urllib.error
import urllib.request

from ..config import llm_base_url, llm_beat_model, llm_fast_model, llm_model, llm_strong_model, require_openai_key


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


def _call_llm(prompt: str, model_name: str) -> str:
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
        with urllib.request.urlopen(request, timeout=180) as response:
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


def call_openai_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_model())


def call_fast_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_fast_model())


def call_beat_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_beat_model())


def call_strong_llm(prompt: str) -> str:
    return _call_llm(prompt, llm_strong_model())
