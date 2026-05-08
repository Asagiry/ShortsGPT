"""Local clipping: ffmpeg subclip, OpenCV face-aware crop, and subtitles."""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from ..config import local_output_dir
from .progress import Progress, user_log

PAUSE_TIGHTEN_VERSION = 7
MIN_VALID_VIDEO_BYTES = 64 * 1024
MIN_TIGHTEN_RANGE_SECONDS = 2.25
MAX_VISUAL_EDGE_KEEP_SECONDS = 4.0
SUBTITLE_X = 540
SUBTITLE_Y = 1120
SUBTITLE_POP_OFFSET = 26
SUBTITLE_POP_MS = 140


def _subprocess_no_window_kwargs() -> Dict:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def _ratio(aspect_ratio: str) -> float:
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _ffmpeg() -> str:
    configured = os.getenv("FFMPEG_BINARY", "").strip()
    if configured and os.path.exists(configured):
        return configured

    found = shutil.which("ffmpeg")
    if found:
        return found

    pattern = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*\bin\ffmpeg.exe"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return "ffmpeg"


def _ffprobe() -> str:
    ffmpeg_path = _ffmpeg()
    candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe" if os.name == "nt" else "ffprobe")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("ffprobe")
    return found or "ffprobe"


def _has_nvenc() -> bool:
    """Check if NVIDIA hardware encoder is available."""
    try:
        result = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
            **_subprocess_no_window_kwargs(),
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _video_codec_args() -> List[str]:
    """Return fastest available video codec args."""
    if _has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]


def _run_ffmpeg(cmd: List[str]) -> None:
    if os.getenv("AI_SHORTS_DEBUG_FFMPEG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[ffmpeg] {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, **_subprocess_no_window_kwargs())
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg was not found. Reinstall FFmpeg or restart Windows so PATH updates.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed with exit code {e.returncode}") from e


def _require_valid_video(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"{label} was not created: {path}")
    size = os.path.getsize(path)
    if size < MIN_VALID_VIDEO_BYTES:
        raise RuntimeError(f"{label} is too small/empty ({size} bytes): {path}")
    try:
        result = subprocess.run(
            [
                _ffprobe(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            **_subprocess_no_window_kwargs(),
        )
        duration = float((result.stdout or "0").strip() or "0")
        if result.returncode != 0 or duration <= 0.2:
            raise RuntimeError(f"{label} has invalid duration ({duration:.2f}s): {path}")
    except FileNotFoundError:
        return
    except ValueError as e:
        raise RuntimeError(f"{label} duration could not be read: {path}") from e


def _merge_short_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged = [(s, e) for s, e in ranges if e > s]
    if len(merged) <= 1:
        return merged
    changed = True
    while changed and len(merged) > 1:
        changed = False
        next_ranges: List[Tuple[float, float]] = []
        index = 0
        while index < len(merged):
            start, end = merged[index]
            if end - start >= MIN_TIGHTEN_RANGE_SECONDS:
                next_ranges.append((start, end))
                index += 1
                continue

            if next_ranges:
                prev_start, _prev_end = next_ranges.pop()
                next_ranges.append((prev_start, end))
            elif index + 1 < len(merged):
                _next_start, next_end = merged[index + 1]
                next_ranges.append((start, next_end))
                index += 1
            else:
                next_ranges.append((start, end))
            changed = True
            index += 1
        merged = next_ranges
    return merged


def _pause_tighten_enabled() -> bool:
    return os.getenv("LOCAL_TIGHTEN_PAUSES", "1").strip().lower() not in {"0", "false", "no", "off"}


def _pause_threshold() -> float:
    try:
        return float(os.getenv("LOCAL_PAUSE_THRESHOLD", "0.65"))
    except ValueError:
        return 0.65


def _pause_keep() -> float:
    try:
        return float(os.getenv("LOCAL_PAUSE_KEEP", "0.22"))
    except ValueError:
        return 0.22


def _speech_ranges_from_transcript(
    transcript: Optional[Dict],
    start_time: float,
    end_time: float,
    tighten_enabled: Optional[bool] = None,
    threshold_override: Optional[float] = None,
    keep_override: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], List[Dict]]:
    if tighten_enabled is None:
        tighten_enabled = _pause_tighten_enabled()
    if not transcript or not tighten_enabled:
        return [(start_time, end_time)], [{
            "source_start": start_time,
            "source_end": end_time,
            "target_start": 0.0,
            "target_end": end_time - start_time,
        }]

    threshold = threshold_override if threshold_override is not None else _pause_threshold()
    keep = keep_override if keep_override is not None else _pause_keep()
    speech = []
    for segment in transcript.get("segments", []):
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end <= start_time or seg_start >= end_time:
            continue
        speech.append((max(start_time, seg_start - 0.08), min(end_time, seg_end + 0.12)))

    if not speech:
        return [(start_time, end_time)], [{
            "source_start": start_time,
            "source_end": end_time,
            "target_start": 0.0,
            "target_end": end_time - start_time,
        }]

    ranges = []
    current_start, current_end = speech[0]
    for seg_start, seg_end in speech[1:]:
        gap = seg_start - current_end
        if gap <= threshold:
            current_end = max(current_end, seg_end)
            continue
        ranges.append((current_start, min(end_time, current_end + keep / 2)))
        current_start = max(start_time, seg_start - keep / 2)
        current_end = seg_end
    ranges.append((current_start, current_end))

    first_start, first_end = ranges[0]
    if first_start - start_time <= MAX_VISUAL_EDGE_KEEP_SECONDS:
        ranges[0] = (start_time, first_end)
    last_start, last_end = ranges[-1]
    if end_time - last_end <= MAX_VISUAL_EDGE_KEEP_SECONDS:
        ranges[-1] = (last_start, end_time)

    cleaned = []
    for start, end in ranges:
        if end - start >= 0.25:
            cleaned.append((max(start_time, start), min(end_time, end)))
    if not cleaned:
        cleaned = [(start_time, end_time)]
    cleaned = _merge_short_ranges(cleaned)

    original_duration = end_time - start_time
    tightened_duration = sum(end - start for start, end in cleaned)
    if len(cleaned) <= 1 or original_duration - tightened_duration < 0.5:
        cleaned = [(start_time, end_time)]

    timing_map = []
    cursor = 0.0
    for source_start, source_end in cleaned:
        duration = source_end - source_start
        timing_map.append({
            "source_start": source_start,
            "source_end": source_end,
            "target_start": cursor,
            "target_end": cursor + duration,
        })
        cursor += duration

    if len(cleaned) > 1:
        user_log(
            "Tightened pauses",
            f"{original_duration:.1f}s -> {cursor:.1f}s across {len(cleaned)} speech ranges",
        )
    return cleaned, timing_map


def _concat_list_path(out_path: str) -> str:
    return out_path + ".concat.txt"


def _cut_tightened_subclip(source_path: str, ranges: List[Tuple[float, float]], out_path: str) -> str:
    if len(ranges) <= 1:
        return _cut_subclip(source_path, ranges[0][0], ranges[0][1], out_path)

    part_paths = []
    list_path = _concat_list_path(out_path)
    try:
        for index, (start, end) in enumerate(ranges, 1):
            part_path = f"{out_path}.part{index:03d}.mp4"
            part_paths.append(part_path)
            _cut_subclip(source_path, start, end, part_path)

        with open(list_path, "w", encoding="utf-8") as fh:
            for part_path in part_paths:
                safe_path = os.path.abspath(part_path).replace("\\", "/").replace("'", "'\\''")
                fh.write(f"file '{safe_path}'\n")

        cmd = [
            _ffmpeg(), "-y", "-loglevel", "warning", "-stats_period", "10", "-stats",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        _run_ffmpeg(cmd)
        _require_valid_video(out_path, "tightened cut clip")
        return out_path
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
        for part_path in part_paths:
            if os.path.exists(part_path):
                os.remove(part_path)


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    user_log("Cutting source", f"{start:.1f}s -> {end:.1f}s")
    duration = max(0.25, end - start)
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "warning", "-stats_period", "10", "-stats",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        *_video_codec_args(),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    _run_ffmpeg(cmd)
    _require_valid_video(out_path, "cut clip")
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15
    frame_index = 0
    progress = Progress("crop frames", total_frames) if total_frames > 0 else None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            if last_center is None:
                last_center = (cx, cy)
            else:
                lx, ly = last_center
                last_center = (
                    int(lx + (cx - lx) * smoothing),
                    int(ly + (cy - ly) * smoothing),
                )
        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        writer.write(frame[y0:y0 + crop_h, x0:x0 + crop_w])
        frame_index += 1
        if progress:
            progress.update(frame_index, f"{frame_index}/{total_frames} frames")

    cap.release()
    writer.release()
    if progress:
        progress.done(f"{frame_index}/{total_frames} frames")

    user_log("Attaching audio")
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "warning", "-stats_period", "10", "-stats",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    try:
        _run_ffmpeg(cmd)
        _require_valid_video(out_path, "vertical clip")
    finally:
        if os.path.exists(silent_path):
            os.remove(silent_path)
    return out_path


def _target_size(aspect_ratio: str) -> Tuple[int, int]:
    ratio = _ratio(aspect_ratio)
    if ratio < 1:
        return 1080, int(round(1080 / ratio))
    return 1080, int(round(1080 / ratio))


def _reframe_fit_blur(in_path: str, out_path: str, aspect_ratio: str) -> str:
    width, height = _target_size(aspect_ratio)
    user_log("Framing", f"fit full frame into {width}x{height}")
    filter_graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=24:2[bg];"
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "warning", "-stats_period", "10", "-stats",
        "-i", in_path,
        "-filter_complex", filter_graph,
        *_video_codec_args(),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd)
    _require_valid_video(out_path, "fit clip")
    return out_path


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    cs = centiseconds % 100
    total_seconds = centiseconds // 100
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    text = " ".join((text or "").split())
    return text.replace("{", "").replace("}", "").replace("\n", " ")


def _ass_text_escape(text: str) -> str:
    return _ass_escape(text).replace("\\", "")


def _filter_path(path: str) -> str:
    value = os.path.abspath(path).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def _map_source_time(timing_map: List[Dict], source_time: float) -> Optional[float]:
    for item in timing_map:
        source_start = float(item["source_start"])
        source_end = float(item["source_end"])
        if source_start <= source_time <= source_end:
            return float(item["target_start"]) + (source_time - source_start)
    return None


def _subtitle_segments(
    transcript: Optional[Dict],
    start_time: float,
    end_time: float,
    timing_map: Optional[List[Dict]] = None,
) -> List[Dict]:
    if not transcript:
        return []

    if timing_map:
        return _subtitle_segments_from_timing_map(transcript, timing_map)

    word_segments = _subtitle_segments_from_words(transcript, start_time, end_time)
    if word_segments:
        return word_segments

    segments = []
    for segment in transcript.get("segments", []):
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end <= start_time or seg_start >= end_time:
            continue
        text = _ass_escape(segment.get("text", ""))
        if not text:
            continue
        clipped_start = max(start_time, seg_start)
        clipped_end = min(end_time, seg_end)
        for sub in _split_subtitle_segment(text, clipped_start, clipped_end):
            segments.append({
                "start": max(0.0, sub["start"] - start_time),
                "end": max(0.25, sub["end"] - start_time),
                "text": sub["text"],
            })
    return segments


def _subtitle_segments_from_timing_map(transcript: Dict, timing_map: List[Dict]) -> List[Dict]:
    word_segments = _subtitle_words_from_timing_map(transcript, timing_map)
    if word_segments:
        return word_segments

    segments = []
    for item in timing_map:
        source_start = float(item["source_start"])
        source_end = float(item["source_end"])
        target_start = float(item["target_start"])
        for segment in transcript.get("segments", []):
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
            if seg_end <= source_start or seg_start >= source_end:
                continue
            text = _ass_escape(segment.get("text", ""))
            if not text:
                continue
            clipped_start = max(source_start, seg_start)
            clipped_end = min(source_end, seg_end)
            mapped_start = target_start + (clipped_start - source_start)
            mapped_end = target_start + (clipped_end - source_start)
            for sub in _split_subtitle_segment(text, mapped_start, mapped_end):
                segments.append({
                    "start": sub["start"],
                    "end": sub["end"],
                    "text": sub["text"],
                })
    return segments


def _subtitle_words_from_timing_map(transcript: Dict, timing_map: List[Dict]) -> List[Dict]:
    words = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []) or []:
            word_start = float(word["start"])
            word_end = float(word["end"])
            text = _ass_text_escape(word.get("text", ""))
            if not text:
                continue
            mapped_start = _map_source_time(timing_map, word_start)
            mapped_end = _map_source_time(timing_map, word_end)
            if mapped_start is None or mapped_end is None:
                continue
            words.append({
                "start": mapped_start,
                "end": max(mapped_start + 0.12, mapped_end),
                "text": text,
            })

    chunks = []
    current = []
    for word in words:
        current.append(word)
        joined = " ".join(w["text"] for w in current)
        sentence_break = bool(re.search(r"[.!?вЂ¦]$", word["text"]))
        if len(current) >= 5 or len(joined) >= 30 or sentence_break:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    return [
        {
            "start": chunk[0]["start"],
            "end": max(chunk[0]["start"] + 0.25, chunk[-1]["end"]),
            "text": _highlight_last_words([w["text"] for w in chunk]),
        }
        for chunk in chunks
        if chunk
    ]


def _subtitle_segments_from_words(transcript: Dict, start_time: float, end_time: float) -> List[Dict]:
    words = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []) or []:
            word_start = float(word["start"])
            word_end = float(word["end"])
            text = _ass_text_escape(word.get("text", ""))
            if not text or word_end <= start_time or word_start >= end_time:
                continue
            words.append({
                "start": max(start_time, word_start),
                "end": min(end_time, word_end),
                "text": text,
            })

    chunks = []
    current = []
    for word in words:
        current.append(word)
        joined = " ".join(w["text"] for w in current)
        sentence_break = bool(re.search(r"[.!?…]$", word["text"]))
        if len(current) >= 5 or len(joined) >= 30 or sentence_break:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    return [
        {
            "start": max(0.0, chunk[0]["start"] - start_time),
            "end": max(0.25, chunk[-1]["end"] - start_time),
            "text": _highlight_last_words([w["text"] for w in chunk]),
        }
        for chunk in chunks
        if chunk
    ]


def _highlight_last_words(words: List[str]) -> str:
    if not words:
        return ""
    split_at = max(0, len(words) - min(2, len(words)))
    normal = " ".join(words[:split_at])
    highlighted = " ".join(words[split_at:])
    color = _random_ass_color(" ".join(words))
    if normal:
        return f"{normal} {{\\c&H{color}&}}{highlighted}{{\\r}}"
    return f"{{\\c&H{color}&}}{highlighted}{{\\r}}"


def _random_ass_color(seed_text: str) -> str:
    palette = [
        "00D7FF",  # orange
        "40FF40",  # green
        "FF66FF",  # pink
        "66FFFF",  # yellow
        "FF9966",  # blue
        "CC66FF",  # purple
        "66CCFF",  # amber
        "99FF99",  # light green
    ]
    digest = hashlib.sha1(seed_text.encode("utf-8", errors="ignore")).digest()
    return palette[digest[0] % len(palette)]


def _split_subtitle_segment(text: str, start: float, end: float) -> List[Dict]:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    for word in words:
        current.append(word)
        joined = " ".join(current)
        sentence_break = bool(re.search(r"[.!?…]$", word))
        if len(current) >= 6 or len(joined) >= 34 or sentence_break:
            chunks.append(joined)
            current = []
    if current:
        chunks.append(" ".join(current))

    duration = max(0.25, end - start)
    total_chars = sum(max(1, len(chunk)) for chunk in chunks)
    cursor = start
    out = []
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            chunk_end = end
        else:
            share = max(1, len(chunk)) / total_chars
            chunk_end = min(end, cursor + duration * share)
        if chunk_end - cursor < 0.35:
            chunk_end = min(end, cursor + 0.35)
        out.append({"start": cursor, "end": chunk_end, "text": chunk})
        cursor = chunk_end
    return out


def _plain_ass_text(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "", text or "").replace(r"\N", " ")


def _ass_subtitle_text(text: str) -> str:
    if "{" not in text:
        return r"\N".join(textwrap.wrap(text, width=24)[:2])
    return text


def _ass_keyword_text(text: str, keywords: Optional[List[str]] = None) -> str:
    if not text or not keywords:
        return _ass_subtitle_text(text)
    escaped = _ass_subtitle_text(_ass_escape(text))
    plain_keywords = sorted(
        {_ass_escape(str(k).strip()) for k in keywords if str(k).strip()},
        key=len,
        reverse=True,
    )[:8]
    for keyword in plain_keywords:
        if len(keyword) < 3:
            continue
        color = _random_ass_color(keyword)
        pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
        escaped = pattern.sub(lambda m: rf"{{\c&H{color}&\bord6}}{m.group(0)}{{\r}}", escaped)
    return escaped


def _subtitle_animation_tags() -> str:
    start_y = SUBTITLE_Y + SUBTITLE_POP_OFFSET
    return (
        rf"\an5\move({SUBTITLE_X},{start_y},{SUBTITLE_X},{SUBTITLE_Y},0,{SUBTITLE_POP_MS})"
        rf"\fscx94\fscy94\t(0,{SUBTITLE_POP_MS},\fscx100\fscy100)"
        r"\fad(35,80)"
    )


def _write_ass(path: str, segments: List[Dict]) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,76,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,6,0,5,60,60,0,1
Style: Intro,Arial,68,&H00FFFFFF,&H0000FFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,5,0,5,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for segment in segments:
        if segment.get("kind") == "intro":
            intro_text = _ass_subtitle_text(_ass_escape(segment["text"]))
            start = _ass_time(segment["start"])
            end = _ass_time(segment["end"])
            tags = (
                rf"\an5\pos({SUBTITLE_X},{SUBTITLE_Y - 250})"
                rf"\fscx92\fscy92\t(0,180,\fscx100\fscy100)"
                r"\fad(90,140)"
            )
            lines.append(f"Dialogue: 2,{start},{end},Intro,,0,0,0,,{{{tags}}}{intro_text}\n")
            continue
        text = _ass_keyword_text(segment["text"], segment.get("highlight_keywords"))
        start = _ass_time(segment["start"])
        end = _ass_time(segment["end"])
        glow_color = _random_ass_color(_plain_ass_text(segment["text"]))
        animation = _subtitle_animation_tags()
        glow = (
            rf"{{{animation}\bord14\blur9\shad0\1a&HFF&\3a&H30&\3c&H{glow_color}&}}"
            f"{text}"
        )
        face = (
            rf"{{{animation}\bord5\blur0.7\shad0\3c&H000000&}}"
            f"{text}"
        )
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{glow}\n"
        )
        lines.append(
            f"Dialogue: 1,{start},{end},Default,,0,0,0,,{face}\n"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _burn_subtitles(video_path: str, segments: List[Dict], highlight: Optional[Dict] = None) -> str:
    highlight = highlight or {}
    intro = str(highlight.get("intro_overlay", "") or "").strip()
    if intro:
        intro = intro[:70]
        segments = [{
            "kind": "intro",
            "start": 0.0,
            "end": min(3.0, max(1.4, segments[0]["start"] + 1.8 if segments else 2.4)),
            "text": intro,
        }] + segments
    keywords = highlight.get("highlight_keywords") if isinstance(highlight.get("highlight_keywords"), list) else []
    if keywords:
        segments = [
            {**segment, "highlight_keywords": keywords}
            if segment.get("kind") != "intro" else segment
            for segment in segments
        ]
    if not segments:
        user_log("Subtitles", "no subtitle lines for this clip")
        return video_path

    user_log("Subtitles", f"burning {len(segments)} lines")
    ass_path = video_path + ".ass"
    subtitled_path = video_path + ".subtitled.mp4"
    _write_ass(ass_path, segments)

    cmd = [
        _ffmpeg(), "-y", "-loglevel", "warning", "-stats_period", "10", "-stats",
        "-i", video_path,
        "-vf", f"ass='{_filter_path(ass_path)}'",
        *_video_codec_args(),
        "-c:a", "copy",
        subtitled_path,
    ]
    try:
        _run_ffmpeg(cmd)
        _require_valid_video(subtitled_path, "subtitled temp clip")
        os.replace(subtitled_path, video_path)
        _require_valid_video(video_path, "subtitled clip")
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
        if os.path.exists(subtitled_path):
            os.remove(subtitled_path)

    return video_path


def _sidecar_path(out_path: str) -> str:
    return out_path + ".json"


def _is_complete_render(out_path: str, highlight: Dict) -> bool:
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 1024 * 1024):
        return False
    sidecar = _sidecar_path(out_path)
    if not os.path.exists(sidecar):
        print(f"[resume] existing clip has no render metadata, will re-render: {out_path}", flush=True)
        return False
    try:
        import json
        with open(sidecar, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        print(f"[resume] existing clip metadata is unreadable, will re-render: {out_path}", flush=True)
        return False
    current_tighten, current_threshold, current_keep = _clip_pause_settings(highlight)
    if data.get("subtitles") is not True or data.get("version") != PAUSE_TIGHTEN_VERSION:
        return False
    if bool(data.get("tighten_pauses")) != current_tighten:
        return False
    if abs(float(data.get("pause_threshold", -1.0)) - current_threshold) > 0.01:
        return False
    if abs(float(data.get("pause_keep", -1.0)) - current_keep) > 0.01:
        return False
    if str(data.get("pause_policy", "")) != str(highlight.get("pause_policy", "")):
        return False
    if str(data.get("intro_overlay", "")) != str(highlight.get("intro_overlay", "")):
        return False
    if data.get("highlight_keywords", []) != highlight.get("highlight_keywords", []):
        return False
    return (
        abs(float(data.get("start_time", -1.0)) - float(highlight.get("start_time", 0.0))) < 0.05
        and abs(float(data.get("end_time", -1.0)) - float(highlight.get("end_time", 0.0))) < 0.05
    )


def _write_render_sidecar(
    out_path: str,
    subtitle_count: int,
    start_time: float,
    end_time: float,
    timing_map: List[Dict],
    highlight: Optional[Dict] = None,
) -> None:
    import json
    highlight = highlight or {}
    tighten_enabled, threshold, keep = _clip_pause_settings(highlight)
    with open(_sidecar_path(out_path), "w", encoding="utf-8") as fh:
        json.dump({
            "version": PAUSE_TIGHTEN_VERSION,
            "subtitles": subtitle_count > 0,
            "subtitle_count": subtitle_count,
            "start_time": start_time,
            "end_time": end_time,
            "tightened": len(timing_map) > 1,
            "tighten_pauses": tighten_enabled,
            "pause_threshold": threshold,
            "pause_keep": keep,
            "pause_policy": highlight.get("pause_policy", ""),
            "intro_overlay": highlight.get("intro_overlay", ""),
            "highlight_keywords": highlight.get("highlight_keywords", []),
            "render_duration": timing_map[-1]["target_end"] if timing_map else end_time - start_time,
        }, fh, indent=2)


def _clip_pause_settings(highlight: Optional[Dict]) -> Tuple[bool, float, float]:
    base_enabled = _pause_tighten_enabled()
    threshold = _pause_threshold()
    keep = _pause_keep()
    policy = str((highlight or {}).get("pause_policy", "")).strip().lower()
    if policy == "keep_reactions":
        return base_enabled, max(threshold, 1.9), max(keep, 0.78)
    if policy == "tight":
        return base_enabled, min(threshold, 0.85), min(keep, 0.28)
    return base_enabled, threshold, keep


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    transcript: Optional[Dict] = None,
    highlight: Optional[Dict] = None,
) -> str:
    cut_path = out_path + ".cut.mp4"
    try:
        tighten_enabled, threshold, keep = _clip_pause_settings(highlight)
        ranges, timing_map = _speech_ranges_from_transcript(
            transcript,
            start_time,
            end_time,
            tighten_enabled=tighten_enabled,
            threshold_override=threshold,
            keep_override=keep,
        )
        _cut_tightened_subclip(source_path, ranges, cut_path)
        frame_mode = os.getenv("LOCAL_FRAME_MODE", "fit").strip().lower()
        user_log("Frame mode", frame_mode)
        if frame_mode in {"crop", "vertical", "face"}:
            _reframe_vertical(cut_path, out_path, aspect_ratio)
        else:
            _reframe_fit_blur(cut_path, out_path, aspect_ratio)
        subtitle_segments = _subtitle_segments(transcript, start_time, end_time, timing_map=timing_map)
        user_log("Subtitle timing", f"{len(subtitle_segments)} lines")
        _burn_subtitles(out_path, subtitle_segments, highlight=highlight)
        _write_render_sidecar(out_path, len(subtitle_segments), start_time, end_time, timing_map, highlight=highlight)
        _require_valid_video(out_path, "final clip")
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript: Optional[Dict] = None,
) -> List[Dict]:
    out_dir = out_dir or local_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = [None] * len(highlights)  # type: ignore

    def _render_one(i: int, h: Dict) -> Tuple[int, Dict]:
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(
            f"\n[studio] Rendering short {i}/{len(highlights)}: {h.get('title', '(untitled)')} "
            f"({float(h.get('start_time', 0.0)):.1f}s->{float(h.get('end_time', 0.0)):.1f}s)",
            flush=True,
        )
        try:
            if _is_complete_render(out_path, h):
                print(f"[resume] short already exists: {out_path}", flush=True)
            else:
                crop_clip_local(
                    source_path,
                    float(h["start_time"]),
                    float(h["end_time"]),
                    aspect_ratio,
                    out_path,
                    transcript=transcript,
                    highlight=h,
                )
            return (i - 1, {**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            return (i - 1, {**h, "clip_url": None, "error": str(e)})

    max_workers = min(len(highlights), os.cpu_count() or 4, 4)
    if max_workers <= 1 or len(highlights) <= 1:
        for i, h in enumerate(highlights, 1):
            idx, result = _render_one(i, h)
            results[idx] = result
    else:
        user_log("Render workers", f"{len(highlights)} shorts in parallel ({max_workers} workers)")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_render_one, i, h) for i, h in enumerate(highlights, 1)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

    return results
