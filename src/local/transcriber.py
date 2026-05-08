"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}.
"""
from typing import Dict, Optional
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import local_whisper_device, local_whisper_model, stt_provider
from .progress import Progress, stage, user_log

_DLL_DIR_HANDLES = []


def _subprocess_no_window_kwargs() -> Dict:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def _configure_native_dll_paths() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    seen = set()
    for entry in list(sys.path):
        root = os.path.abspath(entry)
        candidates = []
        if root.endswith(os.path.join("Lib", "site-packages")):
            candidates.extend([
                os.path.join(root, "ctranslate2"),
                os.path.join(root, "av.libs"),
                os.path.join(root, "numpy.libs"),
                os.path.join(root, "onnxruntime", "capi"),
                os.path.join(root, "torch", "lib"),
            ])
            try:
                candidates.extend(
                    os.path.join(root, name)
                    for name in os.listdir(root)
                    if name.endswith(".libs")
                )
            except OSError:
                pass
        candidates.extend([
            os.path.join(root, "ctranslate2"),
            os.path.join(root, "av.libs"),
            os.path.join(root, "numpy.libs"),
            os.path.join(root, "onnxruntime", "capi"),
        ])
        for relative in (
            os.path.join("..", "ctranslate2"),
            os.path.join("..", "av.libs"),
            os.path.join("..", "numpy.libs"),
            os.path.join("..", "onnxruntime", "capi"),
        ):
            candidates.append(os.path.abspath(os.path.join(root, relative)))
        for path in candidates:
            if path in seen or not os.path.isdir(path):
                continue
            seen.add(path)
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
            except OSError:
                pass


def _resolve_device() -> str:
    configured = local_whisper_device()
    if configured != "auto":
        return configured
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _cpu_count() -> int:
    """Use ALL CPU cores for maximum speed."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_DIR", "") or Path.cwd()).resolve()


def _venv_python() -> Optional[Path]:
    python_path = _project_root() / ".venv" / "Scripts" / "python.exe"
    return python_path if python_path.exists() else None


def _run_transcribe_worker(
    python_path: Path,
    media_path: str,
    language: Optional[str],
    out_json: str,
    force_device: Optional[str] = None,
) -> int:
    env = os.environ.copy()
    env["AI_SHORTS_TRANSCRIBE_WORKER"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(_project_root()) + os.pathsep + env.get("PYTHONPATH", "")
    env["AI_SHORTS_TRANSCRIPT_OUT_JSON"] = out_json
    if force_device:
        env["LOCAL_WHISPER_DEVICE"] = force_device
    cmd = [
        str(python_path),
        "-m",
        "src.local.transcribe_worker",
        media_path,
        out_json,
        language or "",
    ]
    device_note = f" device={force_device}" if force_device else ""
    user_log("Transcriber", f"faster-whisper worker{device_note}")
    process = subprocess.Popen(
        cmd,
        cwd=str(_project_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_no_window_kwargs(),
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return process.wait()


def _transcribe_via_venv_worker(media_path: str, language: Optional[str]) -> Optional[Dict]:
    python_path = _venv_python()
    if not getattr(sys, "frozen", False) or not python_path or os.getenv("AI_SHORTS_TRANSCRIBE_WORKER") == "1":
        return None

    fd, out_json = tempfile.mkstemp(prefix="ai_shorts_transcript_", suffix=".json")
    os.close(fd)
    try:
        code = _run_transcribe_worker(python_path, media_path, language, out_json)
        if code != 0:
            recovered = _read_transcript_json(out_json)
            if recovered and recovered.get("segments"):
                user_log("Transcript recovered", "CUDA worker exited after saving the full transcript; continuing")
                return recovered
            raise RuntimeError(f"transcribe worker failed with exit code {code}")
        with open(out_json, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    finally:
        try:
            os.remove(out_json)
        except OSError:
            pass


def _read_transcript_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_transcript_snapshot(duration: float, segments: list) -> None:
    out_json = os.getenv("AI_SHORTS_TRANSCRIPT_OUT_JSON", "").strip()
    if not out_json:
        return
    tmp_path = out_json + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"duration": duration, "segments": segments}, fh, ensure_ascii=False)
        os.replace(tmp_path, out_json)
    except OSError:
        pass


def _transcribe_in_process(media_path: str, language: Optional[str] = None) -> Dict:
    _configure_native_dll_paths()
    try:
        import av  # noqa: F401
        import ctranslate2  # noqa: F401
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faster-whisper is required but was not importable.\n"
            f"Python executable: {sys.executable}\n"
            f"sys.path: {sys.path[:8]}\n"
            f"Import error: {e!r}\n"
            "Install/check with:\n"
            "    .venv\\Scripts\\pip install faster-whisper"
        ) from e

    device = _resolve_device()
    model_name = local_whisper_model()
    num_cpus = _cpu_count()
    # int8_float32: int8 weights (4x smaller) + float32 compute — fastest on old GPUs
    # int8: fastest on CPU
    compute_type = "int8_float32" if device == "cuda" else "int8"

    word_timestamps = True
    workers = 1 if device == "cuda" else 2
    stage(
        "Transcribing audio",
        f"model={model_name}, device={device}, word timing={'on' if word_timestamps else 'off'}",
    )

    # Suppress ctranslate2 C++ stderr warning about float16 on older GPUs
    import io
    _real_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_cpus,
            num_workers=workers,
        )
    finally:
        sys.stderr = _real_stderr
    segments_iter, info = model.transcribe(
        media_path,
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=word_timestamps,
    )

    segments = []
    duration = float(getattr(info, "duration", 0.0)) or 0.0
    progress = Progress("transcribe", duration) if duration > 0 else None
    for s in segments_iter:
        words = []
        for word in getattr(s, "words", None) or []:
            text = (getattr(word, "word", "") or "").strip()
            if text:
                words.append({
                    "start": float(getattr(word, "start", s.start)),
                    "end": float(getattr(word, "end", s.end)),
                    "text": text,
                })
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
            "words": words,
        })
        if progress:
            progress.update(float(s.end), f"{len(segments)} segments")
        if len(segments) == 1 or len(segments) % 5 == 0:
            _write_transcript_snapshot(duration, segments)

    duration = duration or (segments[-1]["end"] if segments else 0.0)
    _write_transcript_snapshot(duration, segments)
    if progress:
        progress.done(f"{len(segments)} segments")
    else:
        user_log("Transcript ready", f"{len(segments)} speech segments, {duration:.0f}s of audio")
    return {"duration": duration, "segments": segments}


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path."""
    if stt_provider() == "elevenlabs":
        from .elevenlabs_transcriber import transcribe_elevenlabs
        return transcribe_elevenlabs(media_path, language=language)

    worker_result = _transcribe_via_venv_worker(media_path, language)
    if worker_result is not None:
        return worker_result
    return _transcribe_in_process(media_path, language)
