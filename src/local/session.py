import hashlib
import json
import os
import shutil
from typing import Any, Dict, Optional

from ..config import local_output_dir, local_whisper_model


def session_key(youtube_url: str, download_format: str, language: Optional[str]) -> str:
    payload = {
        "url": youtube_url.strip(),
        "format": str(download_format),
        "language": language or "auto",
        "whisper_model": local_whisper_model(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def session_dir(youtube_url: str, download_format: str, language: Optional[str]) -> str:
    path = os.path.join(local_output_dir(), "sessions", session_key(youtube_url, download_format, language))
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _same_source(left: str, right: str) -> bool:
    left_abs = os.path.abspath(left)
    right_abs = os.path.abspath(right)
    if left_abs == right_abs:
        return True
    return os.path.basename(left_abs).lower() == os.path.basename(right_abs).lower()


def hydrate_from_matching_source(current_session_path: str, source_path: str) -> Optional[str]:
    sessions_root = os.path.join(local_output_dir(), "sessions")
    if not os.path.isdir(sessions_root):
        return None

    for name in sorted(os.listdir(sessions_root)):
        candidate_dir = os.path.join(sessions_root, name)
        if not os.path.isdir(candidate_dir) or os.path.abspath(candidate_dir) == os.path.abspath(current_session_path):
            continue

        source_state = read_json(os.path.join(candidate_dir, "source.json"))
        candidate_source = source_state.get("path") if source_state else None
        if not candidate_source or not _same_source(candidate_source, source_path):
            continue

        copied = []
        for filename in (
            "transcript.json",
            "auto_plan.json",
            "analysis_map.json",
            "beat_map.json",
            "edit_plan.json",
            "highlights.json",
            "top.json",
            "verified_top.json",
        ):
            src = os.path.join(candidate_dir, filename)
            dst = os.path.join(current_session_path, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(filename)

        if copied:
            print(f"[resume] reused cached {', '.join(copied)} from session {name}", flush=True)
            return candidate_dir

    return None
