"""Local multimodal analysis for better short selection."""
import math
import os
import struct
import subprocess
import tempfile
import wave
from typing import Dict, List, Tuple

from .clipper import _ffmpeg, _subprocess_no_window_kwargs
from .progress import Progress, stage, user_log


def _scene_cuts(source_path: str, duration: float) -> List[Dict]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * 0.5)))
    prev = None
    cuts = [0.0]
    frame_index = 0
    last_cut = -10.0
    progress = Progress("analyze/video", total_frames) if total_frames > 0 else None
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        small = cv2.resize(frame, (96, 54))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        if prev is not None:
            diff = cv2.absdiff(hsv, prev)
            score = float(diff.mean())
            t = frame_index / fps
            if score >= 27.0 and t - last_cut >= 1.0:
                cuts.append(round(t, 2))
                last_cut = t
        prev = hsv
        if progress:
            progress.update(frame_index, f"{len(cuts)} scene cuts")
        frame_index += 1
    cap.release()
    if progress:
        progress.done(f"{len(cuts)} scene cuts")

    if duration > 0 and (not cuts or abs(cuts[-1] - duration) > 0.5):
        cuts.append(round(duration, 2))
    return [{"time": t} for t in sorted(set(cuts))]


def _extract_mono_wav(source_path: str, wav_path: str) -> bool:
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-i", source_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, **_subprocess_no_window_kwargs())
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024
    except Exception:
        return False


def _audio_energy(source_path: str) -> List[Dict]:
    fd, wav_path = tempfile.mkstemp(prefix="ai_shorts_audio_", suffix=".wav")
    os.close(fd)
    try:
        if not _extract_mono_wav(source_path, wav_path):
            return []
        windows = []
        with wave.open(wav_path, "rb") as wav:
            rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            channels = wav.getnchannels()
            frame_count = wav.getnframes()
            window_frames = max(1, int(rate * 1.0))
            progress = Progress("analyze/audio", frame_count)
            for start_frame in range(0, frame_count, window_frames):
                raw = wav.readframes(window_frames)
                if not raw:
                    break
                if sample_width != 2:
                    continue
                sample_count = len(raw) // 2
                if sample_count <= 0:
                    continue
                samples = struct.unpack("<" + "h" * sample_count, raw)
                if channels > 1:
                    samples = samples[::channels]
                rms = math.sqrt(sum(s * s for s in samples) / max(len(samples), 1)) / 32768.0
                windows.append({
                    "start": round(start_frame / rate, 2),
                    "end": round(min(frame_count, start_frame + window_frames) / rate, 2),
                    "rms": round(rms, 5),
                })
                progress.update(start_frame + window_frames, f"{len(windows)} audio windows")
            progress.done(f"{len(windows)} audio windows")
        if not windows:
            return []
        values = sorted(w["rms"] for w in windows)
        peak_floor = values[int(len(values) * 0.75)] if values else 0.0
        for w in windows:
            w["peak"] = bool(w["rms"] >= peak_floor and w["rms"] > 0.01)
        return windows
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def _utterances(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    utterances = []
    current = []
    for segment in segments:
        if not segment.get("text", "").strip():
            continue
        if current:
            gap = float(segment["start"]) - float(current[-1]["end"])
            text_len = len(" ".join(s.get("text", "") for s in current))
            should_break = gap >= 0.75 or text_len >= 240 or current[-1].get("text", "").strip().endswith((".", "!", "?", "...", "…"))
            if should_break:
                utterances.append(_make_utterance(len(utterances) + 1, current))
                current = []
        current.append(segment)
    if current:
        utterances.append(_make_utterance(len(utterances) + 1, current))
    return utterances


def _make_utterance(index: int, segments: List[Dict]) -> Dict:
    text = " ".join(s.get("text", "").strip() for s in segments).strip()
    return {
        "id": f"u{index:04d}",
        "start": round(float(segments[0]["start"]), 2),
        "end": round(float(segments[-1]["end"]), 2),
        "text": text,
        "word_count": len(text.split()),
    }


def _range_score(start: float, end: float, utterances: List[Dict], audio_windows: List[Dict], scene_cuts: List[Dict]) -> Dict:
    words = [u for u in utterances if float(u["end"]) >= start and float(u["start"]) <= end]
    word_count = sum(int(u.get("word_count", 0)) for u in words)
    duration = max(0.1, end - start)
    density = word_count / duration
    audio = [w for w in audio_windows if float(w["end"]) >= start and float(w["start"]) <= end]
    peak_ratio = sum(1 for w in audio if w.get("peak")) / max(1, len(audio))
    inner_cuts = [
        c for c in scene_cuts
        if start + 1.0 < float(c["time"]) < end - 1.0
    ]
    score = 45.0
    score += min(25.0, density * 7.0)
    score += peak_ratio * 18.0
    score -= min(12.0, len(inner_cuts) * 2.0)
    if 18.0 <= duration <= 45.0:
        score += 8.0
    elif duration > 55.0:
        score -= 5.0
    return {
        "local_score": int(max(0, min(100, round(score)))),
        "word_density": round(density, 2),
        "audio_peak_ratio": round(peak_ratio, 2),
        "scene_cuts_inside": len(inner_cuts),
    }


def _scene_windows(duration: float, utterances: List[Dict], scene_cuts: List[Dict], audio_windows: List[Dict]) -> List[Dict]:
    boundaries = {0.0, round(duration, 2)}
    for cut in scene_cuts:
        t = round(float(cut.get("time", 0.0)), 2)
        if 0.0 <= t <= duration:
            boundaries.add(t)
    for left, right in zip(utterances, utterances[1:]):
        gap = float(right["start"]) - float(left["end"])
        if gap >= 2.2:
            boundaries.add(round((float(left["end"]) + float(right["start"])) / 2.0, 2))
    ordered = sorted(boundaries)
    scenes = []
    for index, (start, end) in enumerate(zip(ordered, ordered[1:]), 1):
        if end - start < 4.0:
            continue
        text_items = [
            str(u.get("text", "")).strip()
            for u in utterances
            if float(u.get("end", 0.0)) >= start and float(u.get("start", 0.0)) <= end
        ]
        if not text_items:
            continue
        audio = [w for w in audio_windows if float(w["end"]) >= start and float(w["start"]) <= end]
        peak_ratio = sum(1 for w in audio if w.get("peak")) / max(1, len(audio))
        text = " ".join(text_items)
        scenes.append({
            "id": f"s{index:04d}",
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "summary_text": text[:420],
            "utterance_count": len(text_items),
            "audio_peak_ratio": round(peak_ratio, 2),
        })
    return scenes


def build_analysis_map(source_path: str, transcript: Dict) -> Dict:
    duration = float(transcript.get("duration", 0.0) or 0.0)
    stage("Analyzing video and audio", "detecting scenes, speech blocks, and loud moments")
    utterances = _utterances(transcript)
    user_log("Speech structure", f"{len(utterances)} dialogue blocks found")
    scenes = _scene_cuts(source_path, duration)
    user_log("Visual structure", f"{max(len(scenes) - 1, 0)} visual cuts found")
    audio = _audio_energy(source_path)
    peak_count = sum(1 for item in audio if item.get("peak"))
    user_log("Audio energy", f"{len(audio)} seconds analyzed, {peak_count} high-energy seconds")
    scene_windows = _scene_windows(duration, utterances, scenes, audio)
    user_log("Scene map", f"{len(scene_windows)} dialogue scenes estimated")
    return {
        "source": "local",
        "duration": duration,
        "utterances": utterances,
        "scene_cuts": scenes,
        "scene_windows": scene_windows,
        "audio_energy": audio,
    }


def score_highlights(highlights: List[Dict], analysis_map: Dict) -> List[Dict]:
    utterances = analysis_map.get("utterances", [])
    audio = analysis_map.get("audio_energy", [])
    scenes = analysis_map.get("scene_cuts", [])
    scored = []
    for highlight in highlights:
        start = float(highlight.get("start_time", 0.0))
        end = float(highlight.get("end_time", 0.0))
        local = _range_score(start, end, utterances, audio, scenes)
        model_score = int(highlight.get("score", 0) or 0)
        combined = round(model_score * 0.72 + local["local_score"] * 0.28)
        scored.append({**highlight, **local, "score": int(combined), "llm_score": model_score})
    return sorted(scored, key=lambda h: int(h.get("score", 0)), reverse=True)
