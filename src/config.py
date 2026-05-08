import os

from dotenv import load_dotenv

load_dotenv()


def llm_api_key() -> str:
    return (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def llm_base_url() -> str:
    return (
        os.getenv("LLM_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )


def llm_model() -> str:
    return (
        os.getenv("LLM_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )


def llm_fast_model() -> str:
    return os.getenv("LLM_FAST_MODEL", "").strip() or llm_model()


def llm_beat_model() -> str:
    return os.getenv("LLM_BEAT_MODEL", "").strip() or llm_fast_model()


def llm_strong_model() -> str:
    return os.getenv("LLM_STRONG_MODEL", "").strip() or llm_model()


def local_whisper_model() -> str:
    return os.getenv("LOCAL_WHISPER_MODEL", "base").strip() or "base"


def local_whisper_device() -> str:
    return os.getenv("LOCAL_WHISPER_DEVICE", "auto").strip() or "auto"


def stt_provider() -> str:
    return os.getenv("STT_PROVIDER", "faster-whisper").strip().lower() or "faster-whisper"


def elevenlabs_api_key() -> str:
    return os.getenv("ELEVENLABS_API_KEY", "").strip()


def edit_profile() -> str:
    return os.getenv("EDIT_PROFILE", "auto").strip().lower() or "auto"


def local_output_dir() -> str:
    return os.getenv("LOCAL_OUTPUT_DIR", "output").strip() or "output"


def require_openai_key() -> str:
    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError(
            "LLM API key is not set. Add LLM_API_KEY/OPENAI_API_KEY or fill it in Settings."
        )
    return api_key
