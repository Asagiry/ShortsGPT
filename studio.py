"""AI Shorts Studio — pywebview desktop app with React UI."""
import io
import json
import importlib
import os
import re
import sys
import threading
import webview
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).resolve().parent
    PROJECT_DIR = EXE_DIR
    if not (PROJECT_DIR / "src").exists():
        PROJECT_DIR = EXE_DIR.parent.parent if (EXE_DIR.parent.parent / "src").exists() else EXE_DIR
    APP_DIR = Path(getattr(sys, "_MEIPASS", EXE_DIR))
else:
    EXE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = EXE_DIR
    APP_DIR = EXE_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

VENV_SITE_PACKAGES = PROJECT_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))
VENV_SCRIPTS = PROJECT_DIR / ".venv" / "Scripts"
if VENV_SCRIPTS.exists():
    os.environ["PATH"] = str(VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
if VENV_SITE_PACKAGES.exists():
    os.environ["PYTHONPATH"] = str(VENV_SITE_PACKAGES) + os.pathsep + os.environ.get("PYTHONPATH", "")

_DLL_DIR_HANDLES = []


def _add_dll_dir(path: Path):
    if not path.exists() or not hasattr(os, "add_dll_directory"):
        return
    value = str(path)
    os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(value))
    except OSError:
        pass


if VENV_SITE_PACKAGES.exists():
    for dll_dir in (
        VENV_SITE_PACKAGES / "ctranslate2",
        VENV_SITE_PACKAGES / "av.libs",
        VENV_SITE_PACKAGES / "numpy.libs",
        VENV_SITE_PACKAGES / "onnxruntime" / "capi",
        VENV_SITE_PACKAGES / "torch" / "lib",
    ):
        _add_dll_dir(dll_dir)
    for dll_dir in VENV_SITE_PACKAGES.glob("*.libs"):
        _add_dll_dir(dll_dir)

if APP_DIR.exists():
    for dll_dir in (
        APP_DIR / "ctranslate2",
        APP_DIR / "av.libs",
        APP_DIR / "numpy.libs",
        APP_DIR / "onnxruntime" / "capi",
    ):
        _add_dll_dir(dll_dir)

UI_DIR = PROJECT_DIR / "ui" / "dist"
if not (UI_DIR / "index.html").exists():
    UI_DIR = APP_DIR / "ui" / "dist"
SETTINGS_FILE = PROJECT_DIR / "gui_settings.json"
ENV_FILE = PROJECT_DIR / ".env"

if os.getenv("AI_SHORTS_SMOKE_IMPORTS") == "1":
    import wave  # noqa: F401
    import urllib.request  # noqa: F401
    import yt_dlp  # noqa: F401
    import importlib as _smoke_importlib
    _smoke_importlib.import_module("src.local.downloader")
    _smoke_importlib.import_module("src.local.media_analysis")
    print("smoke imports ok", flush=True)
    raise SystemExit(0)


class _LogCapture(io.TextIOBase):
    """Intercepts stdout lines and forwards them to the JS frontend."""
    def __init__(self, emit_fn, original):
        self._emit = emit_fn
        self._original = original

    def _timestamped(self, line: str) -> str:
        stripped = line.rstrip()
        if re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s", stripped):
            return stripped
        return f"[{datetime.now().strftime('%H:%M:%S')}] {stripped}"

    def write(self, s):
        if s and s.strip():
            for line in s.splitlines():
                self._emit("log", {"text": self._timestamped(line)})
        return self._original.write(s)

    def flush(self):
        return self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class Api:
    """JS-exposed API for the React frontend."""

    def __init__(self):
        self._window = None
        self._running = False
        self._cancel = False

    def set_window(self, w):
        self._window = w

    # ── Dialog helpers ──────────────────────────────────────────────
    def choose_file(self):
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            directory="",
            file_types=("Video files (*.mp4;*.mkv;*.webm;*.mov;*.avi)", "All files (*.*)"),
            allow_multiple=False,
        )
        return result[0] if result else None

    def choose_files(self):
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            directory="",
            file_types=("Video files (*.mp4;*.mkv;*.webm;*.mov;*.avi)", "All files (*.*)"),
            allow_multiple=True,
        )
        return list(result) if result else []

    def choose_directory(self):
        result = self._window.create_file_dialog(
            webview.FileDialog.FOLDER,
            directory="",
            allow_multiple=False,
        )
        return result[0] if result else None

    # ── Settings persistence ─────────────────────────────────────────
    def _read_env_file(self):
        values = {}
        try:
            lines = ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            return values
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    def load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        env_file = self._read_env_file()
        defaults = {
            "llm_api_key": (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or env_file.get("LLM_API_KEY")
                or env_file.get("OPENAI_API_KEY")
                or ""
            ),
            "llm_base_url": (
                os.environ.get("LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or env_file.get("LLM_BASE_URL")
                or env_file.get("OPENAI_BASE_URL")
                or ""
            ),
            "llm_model": (
                os.environ.get("LLM_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or env_file.get("LLM_MODEL")
                or env_file.get("OPENAI_MODEL")
                or "gpt-4o-mini"
            ),
            "llm_fast_model": (
                os.environ.get("LLM_FAST_MODEL")
                or env_file.get("LLM_FAST_MODEL")
                or ""
            ),
            "llm_beat_model": (
                os.environ.get("LLM_BEAT_MODEL")
                or env_file.get("LLM_BEAT_MODEL")
                or ""
            ),
            "llm_strong_model": (
                os.environ.get("LLM_STRONG_MODEL")
                or env_file.get("LLM_STRONG_MODEL")
                or ""
            ),
        }
        for key, value in defaults.items():
            if not str(data.get(key, "")).strip() and value:
                data[key] = value
        return data

    def save_settings(self, data):
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = SETTINGS_FILE.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(SETTINGS_FILE)
            self._save_llm_env(data)
        except OSError:
            pass

    def _save_llm_env(self, data):
        updates = {
            "LLM_API_KEY": str(data.get("llm_api_key", "")).strip(),
            "LLM_BASE_URL": str(data.get("llm_base_url", "")).strip(),
            "LLM_MODEL": str(data.get("llm_model", "")).strip(),
            "LLM_FAST_MODEL": str(data.get("llm_fast_model", "")).strip(),
            "LLM_BEAT_MODEL": str(data.get("llm_beat_model", "")).strip(),
            "LLM_STRONG_MODEL": str(data.get("llm_strong_model", "")).strip(),
            "OPENAI_API_KEY": str(data.get("llm_api_key", "")).strip(),
            "OPENAI_BASE_URL": str(data.get("llm_base_url", "")).strip(),
            "OPENAI_MODEL": str(data.get("llm_model", "")).strip(),
        }
        existing = {}
        order = []
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    existing[key] = value
                    order.append(key)
                elif line.strip():
                    order.append(line)
        for key, value in updates.items():
            if value:
                existing[key] = value
                if key not in order:
                    order.append(key)

        lines = []
        seen = set()
        for item in order:
            if "=" in item or item.startswith("#"):
                lines.append(item)
                continue
            if item in existing and item not in seen:
                lines.append(f"{item}={existing[item]}")
                seen.add(item)
        for key, value in existing.items():
            if key not in seen:
                lines.append(f"{key}={value}")
        ENV_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # ── Multi-instance support ────────────────────────────────────────
    def _get_env(self, settings):
        """Build env dict for this specific job (no os.environ pollution)."""
        output_dir = str(settings.get("output_dir", "")).strip() or str(PROJECT_DIR / "output")
        env = {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "PROJECT_DIR": str(PROJECT_DIR),
            "LOCAL_WHISPER_MODEL": str(settings.get("whisper", "base")),
            "LOCAL_FRAME_MODE": str(settings.get("frame", "fit")),
            "EDIT_PROFILE": str(settings.get("edit_profile", "auto")),
            "LOCAL_OUTPUT_DIR": output_dir,
        }
        llm_api_key = str(settings.get("llm_api_key", "")).strip()
        llm_base_url = str(settings.get("llm_base_url", "")).strip()
        llm_model = str(settings.get("llm_model", "")).strip()
        llm_fast_model = str(settings.get("llm_fast_model", "")).strip()
        llm_beat_model = str(settings.get("llm_beat_model", "")).strip()
        llm_strong_model = str(settings.get("llm_strong_model", "")).strip()
        env_file = self._read_env_file()
        whisper_device = env_file.get("LOCAL_WHISPER_DEVICE", "").strip()
        stt_provider = env_file.get("STT_PROVIDER", "").strip()
        elevenlabs_api_key = env_file.get("ELEVENLABS_API_KEY", "").strip()
        llm_api_key = llm_api_key or env_file.get("LLM_API_KEY", "") or env_file.get("OPENAI_API_KEY", "")
        llm_base_url = llm_base_url or env_file.get("LLM_BASE_URL", "") or env_file.get("OPENAI_BASE_URL", "")
        llm_model = llm_model or env_file.get("LLM_MODEL", "") or env_file.get("OPENAI_MODEL", "")
        llm_fast_model = llm_fast_model or env_file.get("LLM_FAST_MODEL", "")
        llm_beat_model = llm_beat_model or env_file.get("LLM_BEAT_MODEL", "")
        llm_strong_model = llm_strong_model or env_file.get("LLM_STRONG_MODEL", "")
        if whisper_device:
            env["LOCAL_WHISPER_DEVICE"] = whisper_device
        if stt_provider:
            env["STT_PROVIDER"] = stt_provider
        if elevenlabs_api_key:
            env["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        if llm_api_key:
            env["LLM_API_KEY"] = llm_api_key
            env["OPENAI_API_KEY"] = llm_api_key
        if llm_base_url:
            env["LLM_BASE_URL"] = llm_base_url
            env["OPENAI_BASE_URL"] = llm_base_url
        if llm_model:
            env["LLM_MODEL"] = llm_model
            env["OPENAI_MODEL"] = llm_model
        if llm_fast_model:
            env["LLM_FAST_MODEL"] = llm_fast_model
        if llm_beat_model:
            env["LLM_BEAT_MODEL"] = llm_beat_model
        if llm_strong_model:
            env["LLM_STRONG_MODEL"] = llm_strong_model
        return env

    def _sources_from_settings(self, settings):
        raw_sources = settings.get("sources")
        if isinstance(raw_sources, list):
            sources = [str(s).strip() for s in raw_sources]
        else:
            source = str(settings.get("source", "")).strip()
            sources = [s.strip() for s in source.splitlines()]
        return [s for s in sources if s]

    def _folder_name_for_source(self, source, index):
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            candidate = unquote(Path(parsed.path).stem) or parsed.netloc or f"video_{index:02d}"
        else:
            candidate = Path(source).stem or f"video_{index:02d}"
        candidate = re.sub(r"[<>:\"/\\|?*]+", "_", candidate).strip(" ._")
        return candidate[:80] or f"video_{index:02d}"

    # ── Pipeline control ─────────────────────────────────────────────
    def start_job(self, settings):
        if self._running:
            return {"ok": False, "error": "Job already running"}

        sources = self._sources_from_settings(settings)
        if not sources:
            return {"ok": False, "error": "No source provided"}

        # Set env vars for this job — save/restore so parallel instances don't clash
        job_env = self._get_env(settings)
        saved_env = {k: os.environ.get(k) for k in job_env}
        os.environ.update(job_env)

        num_clips = 0
        clips = str(settings.get("clips", "")).strip()
        if clips and clips != "auto":
            try:
                num_clips = int(clips)
            except ValueError:
                pass

        self._running = True
        self._cancel = False
        self._emit("started", {})

        def _run():
            original_stdout = sys.stdout
            sys.stdout = _LogCapture(self._emit, original_stdout)
            try:
                module_name = "".join(("s", "r", "c"))
                generate_shorts = importlib.import_module(module_name).generate_shorts
                base_output_dir = job_env["LOCAL_OUTPUT_DIR"]
                for index, source in enumerate(sources, 1):
                    if self._cancel:
                        raise RuntimeError("Stopped")
                    item_output_dir = base_output_dir
                    if len(sources) > 1:
                        item_output_dir = str(Path(base_output_dir) / f"{index:02d}_{self._folder_name_for_source(source, index)}")
                    os.environ["LOCAL_OUTPUT_DIR"] = item_output_dir
                    print(f"[studio] batch {index}/{len(sources)} -> {item_output_dir}", flush=True)
                    generate_shorts(
                        video_url=source,
                        num_clips=num_clips,
                        download_format=str(settings.get("quality", "1080")),
                        language=str(settings.get("language", "")) or None,
                    )
                self._emit("done", {"code": 0})
            except Exception as e:
                original_stdout.write(f"[studio] FAILED: {e}\n")
                self._emit("done", {"code": -1, "error": str(e)})
            finally:
                sys.stdout = original_stdout
                # Restore env vars so parallel instances don't clash
                for k, v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                self._running = False
                self._cancel = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"ok": True}

    def stop_job(self):
        self._cancel = True
        self._running = False
        self._emit("done", {"code": 1})

    def open_directory(self, path):
        os.startfile(path)

    # ── Internal ─────────────────────────────────────────────────────
    def _emit(self, event_type, data):
        payload = json.dumps({"type": event_type, **data})
        if self._window:
            try:
                self._window.evaluate_js(f"window.__onStudioEvent && window.__onStudioEvent({payload})")
            except Exception:
                pass


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIR), **kw)

    def log_message(self, format, *args):
        pass  # silence HTTP logs


def _serve_ui():
    """Start a local HTTP server for the React UI assets."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port


def main():
    os.environ["PYTHONUTF8"] = "1"
    api = Api()

    port = _serve_ui()
    url = f"http://127.0.0.1:{port}/index.html"

    window = webview.create_window(
        "AI Shorts Studio",
        url,
        js_api=api,
        width=1280,
        height=860,
        min_size=(1024, 700),
        resizable=True,
        text_select=True,
    )
    api.set_window(window)
    webview.start(debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
