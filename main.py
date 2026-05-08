"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import os
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description="AI Shorts Studio")
    parser.add_argument("url", help="Video URL (YouTube, TikTok, Instagram, Twitter, etc.) or local file path")
    parser.add_argument("--num-clips", type=int, default=0, help="How many shorts to render (default: auto)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium"],
        default=None,
        help="Local faster-whisper model: tiny/base/small/medium. Smaller is faster, larger is more accurate.",
    )
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    args = parser.parse_args()

    if args.whisper_model:
        os.environ["LOCAL_WHISPER_MODEL"] = args.whisper_model

    from src import generate_shorts

    try:
        result = generate_shorts(
            video_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Source video:  {result['source_video_url']}")
    print(f"Highlights:    {len(result['highlights'])} candidates -> kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s -> {s.get('end_time'):.1f}s")
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
