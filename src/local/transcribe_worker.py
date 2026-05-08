import json
import sys

from .transcriber import _transcribe_in_process


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -m src.local.transcribe_worker <media_path> <out_json> [language]", file=sys.stderr)
        return 2

    media_path = sys.argv[1]
    out_json = sys.argv[2]
    language = sys.argv[3].strip() if len(sys.argv) >= 4 else ""
    transcript = _transcribe_in_process(media_path, language or None)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
