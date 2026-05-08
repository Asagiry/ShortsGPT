# ShortsGPT

Desktop app for generating short-form clips from local videos or YouTube URLs.

## Features

- Local video transcription with faster-whisper.
- OpenAI-compatible LLM settings: API key, endpoint, model name.
- AUTO clip count based on actual strong moments, not video length.
- Batch processing for multiple source files.
- Local video analysis for speech blocks, audio energy, and scene flow.
- Pause tightening with safeguards for visual/reaction endings.
- Thin desktop build via PyInstaller and React UI.

## Development

Install Python dependencies:

```powershell
.venv\Scripts\pip install -r requirements.txt
```

Install UI dependencies and build:

```powershell
cd ui
npm.cmd install
npm.cmd run build
```

Run from source:

```powershell
.venv\Scripts\python.exe studio.py
```

## Local Settings

Runtime settings are saved locally in `gui_settings.json` and `.env`. These files are ignored by git because they can contain API keys.
