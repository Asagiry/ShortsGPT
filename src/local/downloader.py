"""Local video input handling.

Accepts local files, direct media URLs, or URLs supported by yt-dlp.
Returns a local media path so the rest of the local pipeline can read it.
"""
import os
import shutil
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

from ..config import local_output_dir
from .progress import Progress

MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for YouTube downloads but was not importable.\n"
            f"Import error: {e!r}\n"
            "Install/check with:\n"
            "    .venv\\Scripts\\pip install -r requirements.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={height}][ext=mp4]/best"
    )


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _reuse_existing_source(out_dir: str) -> Optional[str]:
    for name in sorted(os.listdir(out_dir)):
        if not name.startswith("source_"):
            continue
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS:
            if os.path.getsize(path) > 1024 * 1024:
                print(f"[resume] reusing existing source: {path}", flush=True)
                return path
    return None


def _media_extension(value: str) -> str:
    path = unquote(urlparse(value).path if _is_url(value) else value)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    return ext if ext in MEDIA_EXTENSIONS else ".mp4"


def _looks_like_direct_media_url(value: str) -> bool:
    return _media_extension(value) in MEDIA_EXTENSIONS and os.path.splitext(unquote(urlparse(value).path))[1].lower() in MEDIA_EXTENSIONS


def _copy_local_file(source_path: str, out_dir: str) -> str:
    if not os.path.isfile(source_path):
        for ext in MEDIA_EXTENSIONS:
            candidate = source_path + ext
            if os.path.isfile(candidate):
                print(f"[input/local] resolved missing extension: {candidate}", flush=True)
                source_path = candidate
                break
    if not os.path.isfile(source_path):
        raise RuntimeError(f"local input file not found: {source_path}")

    ext = _media_extension(source_path)
    out_path = os.path.join(out_dir, f"source_local{ext}")
    if os.path.abspath(source_path) != os.path.abspath(out_path):
        shutil.copy2(source_path, out_path)

    print(f"[input/local] ready: {out_path}", flush=True)
    return out_path


def _download_direct_media_url(video_url: str, out_dir: str) -> str:
    import requests

    ext = _media_extension(video_url)
    out_path = os.path.join(out_dir, f"source_direct{ext}")
    print(f"[download/direct] {video_url} -> {out_path}", flush=True)

    with requests.get(video_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        progress = Progress("download", total) if total > 0 else None
        downloaded = 0
        with open(out_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress.update(downloaded, f"{downloaded // (1024 * 1024)} MB")
        if progress:
            progress.done(f"{downloaded // (1024 * 1024)} MB")

    print(f"[download/direct] ready: {out_path}", flush=True)
    return out_path


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str:
    """Resolve the source video to a local media path."""
    out_dir = out_dir or local_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    if not _is_url(video_url):
        return _copy_local_file(video_url, out_dir)

    if _looks_like_direct_media_url(video_url):
        return _download_direct_media_url(video_url, out_dir)

    existing = _reuse_existing_source(out_dir)
    if existing:
        return existing

    yt_dlp = _import_ytdlp()
    print(f"[download/local] {video_url} @ {fmt}p -> {out_dir}/", flush=True)
    progress_holder = {"bar": None}

    def progress_hook(status: Dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            downloaded = status.get("downloaded_bytes") or 0
            if total:
                if progress_holder["bar"] is None:
                    progress_holder["bar"] = Progress("yt-dlp", total)
                progress_holder["bar"].update(downloaded, f"{downloaded // (1024 * 1024)} MB")
        elif status.get("status") == "finished" and progress_holder["bar"] is not None:
            progress_holder["bar"].done("downloaded")

    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": os.path.join(out_dir, "source_%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        path = ydl.prepare_filename(info)
        if not os.path.exists(path):
            stem, _ = os.path.splitext(path)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    path = stem + ext
                    break

    print(f"[download/local] ready: {path}", flush=True)
    return path
