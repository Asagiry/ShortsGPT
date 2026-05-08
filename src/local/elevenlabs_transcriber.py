"""Optional ElevenLabs Scribe transcription with word-level timestamps."""
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..config import elevenlabs_api_key
from .progress import stage


def _word_items(words: List[Dict]) -> List[Dict]:
    items = []
    for word in words or []:
        if word.get("type") != "word":
            continue
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        items.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker_id": word.get("speaker_id"),
        })
    return items


def _segments_from_words(words: List[Dict]) -> List[Dict]:
    segments = []
    current: List[Dict] = []
    for word in words:
        gap = word["start"] - current[-1]["end"] if current else 0.0
        joined = " ".join(w["text"] for w in current)
        should_break = (
            bool(current)
            and (
                gap >= 0.8
                or len(current) >= 26
                or len(joined) >= 180
                or current[-1]["text"].endswith((".", "!", "?", "...", "…"))
            )
        )
        if should_break:
            segments.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(w["text"] for w in current).strip(),
                "words": [{k: v for k, v in w.items() if k != "speaker_id"} for w in current],
            })
            current = []
        current.append(word)

    if current:
        segments.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(w["text"] for w in current).strip(),
            "words": [{k: v for k, v in w.items() if k != "speaker_id"} for w in current],
        })
    return segments


def transcribe_elevenlabs(media_path: str, language: Optional[str] = None) -> Dict:
    api_key = elevenlabs_api_key()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")

    path = Path(media_path)
    if not path.exists():
        raise RuntimeError(f"local input file not found: {media_path}")

    stage("Transcribing audio", "ElevenLabs Scribe v2 word timestamps")
    data = {
        "model_id": "scribe_v2",
        "timestamps_granularity": "word",
        "tag_audio_events": "false",
        "diarize": "false",
        "no_verbatim": "false",
    }
    if language:
        data["language_code"] = language

    with path.open("rb") as fh:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            data=data,
            files={"file": (path.name, fh, "application/octet-stream")},
            timeout=3600,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs STT failed: HTTP {response.status_code}: {response.text[:1000]}")

    payload = response.json()
    words = _word_items(payload.get("words", []))
    segments = _segments_from_words(words)
    duration = words[-1]["end"] if words else 0.0
    print(
        f"[transcribe/elevenlabs] {len(segments)} segments, {len(words)} words, {duration:.0f}s of audio",
        flush=True,
    )
    return {
        "duration": duration,
        "segments": segments,
        "provider": "elevenlabs",
        "language_code": payload.get("language_code"),
        "text": payload.get("text", ""),
    }
