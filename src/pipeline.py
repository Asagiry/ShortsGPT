"""End-to-end orchestrator — local mode only.

Pipeline: yt-dlp download -> faster-whisper transcribe -> OpenAI LLM highlights -> ffmpeg/opencv render.
All processing runs locally. OPENAI_API_KEY required for highlight analysis.
"""
import os
import re
import shutil
import html
from typing import Dict, List, Optional

from .config import edit_profile, local_output_dir
from .highlights import build_beat_map, build_episode_digest, decide_auto_clip_count, decide_edit_plan, final_quality_review_with_llm, get_highlights, keep_postable_highlights, refine_highlight_boundaries_with_llm


def _snap_highlights_to_segments(highlights: List[Dict], transcript: Dict) -> List[Dict]:
    """Expand highlight times to full phrase boundaries."""
    segments = transcript.get("segments", [])
    if not segments:
        return highlights

    snapped = []
    duration = float(transcript.get("duration", 0.0) or 0.0)
    for highlight in highlights:
        start = float(highlight["start_time"])
        end = float(highlight["end_time"])
        original_end = end
        nearby = [
            s for s in segments
            if float(s["end"]) >= start - 2.0 and float(s["start"]) <= end + 2.0
        ]
        if nearby:
            start = max(0.0, float(nearby[0]["start"]) - 0.25)
            end = _extend_to_natural_ending(segments, nearby[-1], original_end, duration)
        snapped.append({**highlight, "start_time": start, "end_time": end})
    return snapped


def _segment_index_at(segments: List[Dict], time_value: float) -> Optional[int]:
    for index, segment in enumerate(segments):
        if float(segment["start"]) < time_value < float(segment["end"]):
            return index
    return None


def _extend_to_natural_ending(segments: List[Dict], last_segment: Dict, original_end: float, duration: float) -> float:
    max_end = original_end + 12.0
    last_index = segments.index(last_segment)
    candidate_end = float(last_segment["end"])

    for index, segment in enumerate(segments[last_index:], last_index):
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        text = (segment.get("text") or "").strip()
        if seg_start > max_end:
            break

        candidate_end = seg_end
        next_start = float(segments[index + 1]["start"]) if index + 1 < len(segments) else duration
        pause_after = next_start - seg_end
        if text.endswith((".", "!", "?", "...", "…")) or pause_after >= 0.8:
            break

    if duration > 0:
        candidate_end = min(candidate_end, duration)
    return candidate_end + 0.45


TARGET_CLIP_SECONDS = 60.0
MAX_CLIP_SECONDS = 75.0
BOUNDARY_REVIEW_VERSION = 2
EPISODE_DIGEST_VERSION = 1
FINAL_QUALITY_VERSION = 1


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[\wЁёА-Яа-я]+", (text or "").casefold())


def _transcript_words(transcript: Dict) -> List[Dict]:
    out = []
    for segment in transcript.get("segments", []):
        words = segment.get("words") or []
        if words:
            for word in words:
                text = str(word.get("text", "")).strip()
                norm = _norm_words(text)
                if not norm:
                    continue
                out.append({
                    "text": text,
                    "norm": norm[0],
                    "start": float(word.get("start", segment.get("start", 0.0))),
                    "end": float(word.get("end", segment.get("end", 0.0))),
                })
        else:
            text_words = _norm_words(str(segment.get("text", "")))
            if not text_words:
                continue
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            span = max(0.1, seg_end - seg_start)
            for index, word in enumerate(text_words):
                out.append({
                    "text": word,
                    "norm": word,
                    "start": seg_start + span * index / max(1, len(text_words)),
                    "end": seg_start + span * (index + 1) / max(1, len(text_words)),
                })
    return out


def _find_anchor_time(words: List[Dict], anchor: str, start_hint: float, end_hint: float, prefer_end: bool) -> Optional[float]:
    query = _norm_words(anchor)
    if not query:
        return None
    window_start = max(0.0, start_hint - 45.0)
    window_end = end_hint + 45.0
    best = None
    best_score = -1.0
    q_len = len(query)
    for i in range(0, max(0, len(words) - q_len + 1)):
        if words[i]["start"] < window_start or words[i]["end"] > window_end:
            continue
        slice_words = [w["norm"] for w in words[i:i + q_len]]
        exact = slice_words == query
        if exact:
            score = 1000.0 - abs((words[i + q_len - 1]["end"] if prefer_end else words[i]["start"]) - (end_hint if prefer_end else start_hint))
        else:
            overlap = sum(1 for a, b in zip(slice_words, query) if a == b)
            score = overlap / max(q_len, 1)
            if score < 0.72:
                continue
            score -= abs((words[i + q_len - 1]["end"] if prefer_end else words[i]["start"]) - (end_hint if prefer_end else start_hint)) / 200.0
        if score > best_score:
            best_score = score
            best = words[i + q_len - 1]["end"] if prefer_end else words[i]["start"]
    return best


def _apply_text_anchor_boundaries(highlights: List[Dict], transcript: Dict) -> List[Dict]:
    words = _transcript_words(transcript)
    if not words:
        return highlights
    duration = float(transcript.get("duration", 0.0) or 0.0)
    fixed = []
    for h in highlights:
        start = float(h.get("start_time", 0.0))
        end = float(h.get("end_time", 0.0))
        start_anchor = str(h.get("start_anchor_text", "")).strip()
        end_anchor = str(h.get("end_anchor_text", "")).strip()
        mapped_start = _find_anchor_time(words, start_anchor, start, end, prefer_end=False) if start_anchor else None
        mapped_end = _find_anchor_time(words, end_anchor, start, end, prefer_end=True) if end_anchor else None
        if mapped_start is not None:
            start = max(0.0, mapped_start - 0.15)
        if mapped_end is not None:
            try:
                tail = max(0.0, min(float(h.get("include_after_end_seconds", 0.0) or 0.0), 3.0))
            except (TypeError, ValueError):
                tail = 0.0
            end = mapped_end + 0.25 + tail
        if duration > 0:
            end = min(end, duration)
        if end > start:
            fixed.append({**h, "start_time": start, "end_time": end})
    return fixed


def _has_natural_pause_after(segments: List[Dict], index: int, duration: float) -> bool:
    segment = segments[index]
    seg_end = float(segment["end"])
    next_start = float(segments[index + 1]["start"]) if index + 1 < len(segments) else duration
    return next_start - seg_end >= 0.55


def _ends_like_sentence(text: str) -> bool:
    return (text or "").strip().endswith((".", "!", "?", "...", "…", "вЂ¦"))


def _clip_limit_seconds() -> float:
    try:
        configured = float(os.getenv("LOCAL_MAX_CLIP_SECONDS", str(MAX_CLIP_SECONDS)))
    except ValueError:
        configured = MAX_CLIP_SECONDS
    return max(TARGET_CLIP_SECONDS, min(configured, 180.0))


def _word_boundary_end(segments: List[Dict], start: float, end: float, max_end: float, duration: float) -> float:
    """Move an end timestamp out of the middle of a word when word timestamps exist."""
    best_end = end
    for segment in segments:
        if float(segment.get("end", 0.0)) < start or float(segment.get("start", 0.0)) > max_end:
            continue
        for word in segment.get("words", []) or []:
            try:
                word_start = float(word.get("start", 0.0))
                word_end = float(word.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if word_start < end < word_end + 0.05:
                candidate = word_end + 0.12
                if candidate <= max_end:
                    best_end = max(best_end, candidate)
            elif 0.0 <= word_start - end <= 0.25 and word_end <= max_end:
                best_end = max(best_end, word_end + 0.12)
            if best_end > end:
                break
        if best_end > end:
            break
    if duration > 0:
        best_end = min(best_end, duration)
    return best_end


def _shift_start_to_fit(start: float, end: float, segments: List[Dict], limit: float) -> Optional[float]:
    if end - start <= limit:
        return start
    earliest_allowed = end - limit
    candidates = [
        max(0.0, float(segment["start"]) - 0.2)
        for segment in segments
        if float(segment["start"]) >= earliest_allowed and end - float(segment["start"]) >= 10.0
    ]
    return candidates[0] if candidates else None


def _extend_to_min_duration_clean(start: float, end: float, segments: List[Dict], duration: float, max_end: float) -> float:
    if end - start >= 10.0:
        return end
    for index, segment in enumerate(segments):
        seg_end = float(segment["end"])
        if seg_end <= end:
            continue
        if seg_end > max_end:
            break
        end = seg_end + 0.25
        if end - start >= 10.0 and (
            _ends_like_sentence(segment.get("text", "")) or _has_natural_pause_after(segments, index, duration)
        ):
            break
    return min(end, duration) if duration > 0 else end


def _extend_dialogue_tail(segments: List[Dict], start: float, end: float, max_end: float, duration: float) -> float:
    tail_limit = min(max_end, end + 8.0)
    current_end = end
    for index, segment in enumerate(segments):
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end <= current_end + 0.05:
            continue
        if seg_start > current_end + 1.15 or seg_end > tail_limit:
            break
        if seg_end - start < 10.0:
            current_end = seg_end + 0.25
            continue
        current_end = seg_end + 0.35
        next_start = float(segments[index + 1]["start"]) if index + 1 < len(segments) else duration
        if next_start - seg_end >= 0.95:
            break
        if current_end - end >= 4.0 and _ends_like_sentence(segment.get("text", "")):
            break
    if duration > 0:
        current_end = min(current_end, duration)
    return current_end


def _natural_end_before_cap(start: float, end: float, transcript: Dict) -> Optional[float]:
    segments = transcript.get("segments", [])
    limit = _clip_limit_seconds()
    if end - start <= limit or not segments:
        return end

    max_end = start + limit
    duration = float(transcript.get("duration", 0.0) or 0.0)
    eligible = [
        (index, segment)
        for index, segment in enumerate(segments)
        if float(segment["end"]) <= max_end and float(segment["end"]) > start + 10.0
    ]
    if not eligible:
        return None

    natural = [
        float(segment["end"])
        for index, segment in eligible
        if _ends_like_sentence(segment.get("text", "")) or _has_natural_pause_after(segments, index, duration)
    ]
    if natural:
        return min(max(natural) + 0.25, max_end)
    return min(float(eligible[-1][1]["end"]) + 0.25, max_end)


def _complete_clip_boundaries(highlights: List[Dict], transcript: Dict) -> List[Dict]:
    """Final deterministic guard: no clip should end inside a spoken segment."""
    segments = transcript.get("segments", [])
    if not segments:
        return highlights

    limit = _clip_limit_seconds()
    duration = float(transcript.get("duration", 0.0) or 0.0)
    fixed = []
    for highlight in highlights:
        start = max(0.0, float(highlight.get("start_time", 0.0)))
        end = float(highlight.get("end_time", 0.0))
        max_end = start + limit

        containing = _segment_index_at(segments, end)
        if containing is not None:
            seg_end = float(segments[containing]["end"])
            if seg_end <= max_end:
                end = seg_end + 0.25
            else:
                shifted_start = _shift_start_to_fit(start, seg_end + 0.25, segments, limit)
                if shifted_start is None:
                    continue
                start = shifted_start
                end = seg_end + 0.25
                max_end = start + limit

        end_index = None
        for index, segment in enumerate(segments):
            if abs(float(segment["end"]) - min(end, float(segment["end"]))) < 0.35 and float(segment["end"]) <= end + 0.35:
                if float(segment["end"]) >= start:
                    end_index = index

        if end_index is not None:
            for index in range(end_index, len(segments)):
                segment = segments[index]
                seg_start = float(segment["start"])
                seg_end = float(segment["end"])
                if seg_start > end + 1.5 or seg_end > max_end:
                    break
                end = seg_end + 0.25
                if _ends_like_sentence(segment.get("text", "")) or _has_natural_pause_after(segments, index, duration):
                    break

        end = _word_boundary_end(segments, start, end, max_end, duration)
        end = _extend_dialogue_tail(segments, start, end, max_end, duration)
        if duration > 0:
            end = min(end, duration)
        if end - start > limit:
            shifted_start = _shift_start_to_fit(start, end, segments, limit)
            if shifted_start is None:
                continue
            start = shifted_start
            max_end = start + limit
        if end - start < 10.0:
            end = _extend_to_min_duration_clean(start, end, segments, duration, max_end)
        if end - start < 10.0 or end - start > limit + 0.25:
            continue
        fixed.append({**highlight, "start_time": start, "end_time": end})
    return fixed


def _snap_beat_boundaries_to_utterances(beat_map: Dict, analysis_map: Dict) -> Dict:
    utterances = {str(u.get("id")): u for u in analysis_map.get("utterances", [])}
    beats = []
    for beat in beat_map.get("beats", []):
        start_id = str(beat.get("utterance_start_id", ""))
        end_id = str(beat.get("utterance_end_id", ""))
        updated = dict(beat)
        if start_id in utterances:
            updated["start_time"] = float(utterances[start_id]["start"])
        if end_id in utterances:
            updated["end_time"] = float(utterances[end_id]["end"])
        beats.append(updated)
    return {**beat_map, "beats": beats}


def _enforce_duration_cap(highlights: List[Dict], transcript: Dict) -> List[Dict]:
    """Keep clips within the render limit without cutting off their endings."""
    capped = []
    segments = transcript.get("segments", [])
    limit = _clip_limit_seconds()
    for h in highlights:
        start = float(h.get("start_time", 0))
        end = float(h.get("end_time", 0))
        if end - start > limit:
            shifted_start = _shift_start_to_fit(start, end, segments, limit) if segments else None
            if shifted_start is None:
                continue
            h = {**h, "start_time": shifted_start}
        capped.append(h)
    return _complete_clip_boundaries(capped, transcript)


def _write_upload_metadata(output_dir: str, shorts: List[Dict]) -> str:
    from .local.session import write_json

    metadata = []
    for index, short in enumerate(shorts, 1):
        clip_path = short.get("clip_url")
        metadata.append({
            "index": index,
            "file": clip_path,
            "title": short.get("title", f"short_{index:02d}"),
            "titles": short.get("titles", []),
            "description": short.get("description", ""),
            "hashtags": short.get("hashtags", []),
            "pinned_comment": short.get("pinned_comment", ""),
            "viral_score": short.get("viral_score", short.get("score", 0)),
            "score_matrix": short.get("score_matrix", {}),
            "hook_sentence": short.get("hook_sentence", ""),
            "intro_overlay": short.get("intro_overlay", ""),
            "semantic_key": short.get("semantic_key", ""),
            "reason": short.get("virality_reason", ""),
            "start_time": short.get("start_time"),
            "end_time": short.get("end_time"),
            "error": short.get("error"),
        })
    path = os.path.join(output_dir, "upload_metadata.json")
    write_json(path, {"clips": metadata})
    return path


def _write_contact_sheet(output_dir: str, shorts: List[Dict]) -> str:
    rows = []
    for index, short in enumerate(shorts, 1):
        clip_path = short.get("clip_url") or ""
        rel_clip = os.path.basename(clip_path) if clip_path else ""
        titles = short.get("titles") if isinstance(short.get("titles"), list) else []
        hashtags = short.get("hashtags") if isinstance(short.get("hashtags"), list) else []
        matrix = short.get("score_matrix") if isinstance(short.get("score_matrix"), dict) else {}
        rows.append(f"""
<section class="clip">
  <video src="{html.escape(rel_clip)}" controls preload="metadata"></video>
  <div class="meta">
    <h2>{index:02d}. {html.escape(str(short.get('title', 'Untitled')))}</h2>
    <p><b>Score:</b> {html.escape(str(short.get('viral_score', short.get('score', 0))))}</p>
    <p><b>Hook:</b> {html.escape(str(short.get('hook_sentence', '')))}</p>
    <p><b>Overlay:</b> {html.escape(str(short.get('intro_overlay', '')))}</p>
    <p><b>Why:</b> {html.escape(str(short.get('virality_reason', '')))}</p>
    <p><b>Titles:</b> {html.escape(' | '.join(str(t) for t in titles))}</p>
    <p><b>Description:</b> {html.escape(str(short.get('description', '')))}</p>
    <p><b>Hashtags:</b> {html.escape(' '.join(str(t) for t in hashtags))}</p>
    <p><b>Pinned:</b> {html.escape(str(short.get('pinned_comment', '')))}</p>
    <p><b>Matrix:</b> {html.escape(json_dumps_compact(matrix))}</p>
  </div>
</section>""")
    path = os.path.join(output_dir, "clips_review.html")
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Shorts Review</title>
<style>
body{{font-family:Arial,sans-serif;background:#111;color:#eee;margin:24px}}
.clip{{display:grid;grid-template-columns:260px 1fr;gap:18px;margin:0 0 24px;padding:16px;border:1px solid #333;border-radius:8px;background:#181818}}
video{{width:260px;max-height:460px;background:#000}}
h2{{margin:0 0 10px;font-size:20px}} p{{margin:6px 0;line-height:1.35}} b{{color:#9ee}}
@media(max-width:760px){{.clip{{grid-template-columns:1fr}}video{{width:100%;max-height:none}}}}
</style></head><body>
<h1>Shorts Review</h1>
{''.join(rows)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return path


def json_dumps_compact(data: Dict) -> str:
    import json
    return json.dumps(data or {}, ensure_ascii=False, separators=(",", ":"))


def generate_shorts(
    video_url: str,
    num_clips: int = 0,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
) -> Dict:
    """Run the full local pipeline and return a structured result.

    Args:
        video_url: source URL (any yt-dlp supported site) or local file path.
        num_clips: how many shorts to render (0 = auto).
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.

    Returns:
        {
          "source_video_url": str,
          "transcript": {...},
          "highlights": [...],
          "shorts": [...],
        }
    """
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_beat_llm, call_fast_llm
    from .local.media_analysis import build_analysis_map, score_highlights
    from .local.progress import stage, user_log
    from .local.session import hydrate_from_matching_source, hydrate_latest_transcript, read_json, session_dir, write_json
    from .local.transcriber import transcribe_local

    output_dir = local_output_dir()
    session_path = session_dir(video_url, download_format, language)
    source_json = f"{session_path}/source.json"
    transcript_json = f"{session_path}/transcript.json"
    analysis_map_json = f"{session_path}/analysis_map.json"
    episode_digest_json = f"{session_path}/episode_digest.json"
    beat_map_json = f"{session_path}/beat_map.json"
    highlights_json = f"{session_path}/highlights.json"
    auto_plan_json = f"{session_path}/auto_plan.json"
    edit_plan_json = f"{session_path}/edit_plan.json"
    top_json = f"{session_path}/top.json"
    verified_top_json = f"{session_path}/verified_top.json"
    final_quality_json = f"{session_path}/final_quality.json"
    result_json = f"{session_path}/result.json"
    user_log("Job folder", output_dir)

    stage("Preparing source", "checking local file or downloading video")
    source_state = read_json(source_json)
    if (
        source_state
        and source_state.get("path")
        and os.path.exists(source_state["path"])
        and os.path.getsize(source_state["path"]) > 1024 * 1024
    ):
        source_path = source_state["path"]
        user_log("Source ready", os.path.basename(source_path))
    else:
        source_path = download_youtube_local(video_url, fmt=download_format)
        write_json(source_json, {"path": source_path})

    hydrated_session = hydrate_from_matching_source(session_path, source_path)
    if not hydrated_session:
        hydrate_latest_transcript(session_path)

    transcript = read_json(transcript_json)
    if transcript:
        user_log("Transcript ready", f"{len(transcript.get('segments', []))} speech segments loaded from cache")
    else:
        transcript = transcribe_local(source_path, language=language)
        write_json(transcript_json, transcript)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    analysis_map = read_json(analysis_map_json)
    if analysis_map and analysis_map.get("utterances") and analysis_map.get("scene_windows"):
        user_log(
            "Media analysis ready",
            f"{len(analysis_map.get('utterances', []))} dialogue blocks, "
            f"{max(len(analysis_map.get('scene_cuts', [])) - 1, 0)} visual cuts",
        )
    else:
        analysis_map = build_analysis_map(source_path, transcript)
        write_json(analysis_map_json, analysis_map)

    episode_digest = read_json(episode_digest_json)
    if episode_digest and episode_digest.get("complete") and episode_digest.get("version") == EPISODE_DIGEST_VERSION:
        user_log("Episode brief ready", str(episode_digest.get("logline", "loaded from cache")))
    else:
        stage("Understanding episode", "building story brief for smarter clip selection")
        episode_digest = build_episode_digest(
            transcript,
            call_fast_llm,
            analysis_map=analysis_map,
            cache_path=episode_digest_json,
        )
        episode_digest["version"] = EPISODE_DIGEST_VERSION
        write_json(episode_digest_json, episode_digest)
        user_log("Episode brief ready", str(episode_digest.get("logline", "")))

    requested_edit_profile = edit_profile()
    stage("Choosing edit style", f"profile setting: {requested_edit_profile}")
    edit_plan = read_json(edit_plan_json)
    if not edit_plan or edit_plan.get("requested_profile") != requested_edit_profile:
        edit_plan = decide_edit_plan(transcript, call_fast_llm, requested_profile=requested_edit_profile)
        edit_plan["requested_profile"] = requested_edit_profile
        write_json(edit_plan_json, edit_plan)
    pause_mode = "pause cuts on" if edit_plan.get("tighten_pauses") else "pause cuts off"
    user_log(
        "Edit style selected",
        f"{edit_plan.get('profile')} ({pause_mode}, gap>{float(edit_plan.get('pause_threshold', 0)):.2f}s, keep {float(edit_plan.get('pause_keep', 0)):.2f}s)",
    )
    user_log("Why", str(edit_plan.get("reason", "")))

    if num_clips <= 0:
        auto_plan = read_json(auto_plan_json)
        cached_auto_count = int(auto_plan.get("num_clips", 0) or 0) if auto_plan else 0
        if auto_plan and auto_plan.get("source") == "llm" and "num_clips" in auto_plan:
            num_clips = cached_auto_count
            user_log("Clip count", f"{num_clips} shorts loaded from cache")
        else:
            stage("Planning clip count", "OpenAI LLM decides AUTO")
            auto_plan = decide_auto_clip_count(transcript, call_fast_llm)
            num_clips = int(auto_plan["num_clips"])
            write_json(auto_plan_json, auto_plan)
            user_log("Clip count", f"{num_clips} shorts planned")
            user_log("Why", str(auto_plan.get("reason", "")))
        if num_clips <= 0:
            user_log("No strong clips", "LLM found no genuinely postable moments in this video")
            result = {
                "source_video_url": source_path,
                "transcript": transcript,
                "highlights": [],
                "shorts": [],
            }
            write_json(result_json, result)
            return result

    stage("GPT story mapping", "reading transcript chunks and finding complete jokes/scenes/ideas")
    beat_map = read_json(beat_map_json)
    if beat_map and beat_map.get("complete") and beat_map.get("beats"):
        user_log("Story beats ready", f"{len(beat_map.get('beats', []))} beats loaded from cache")
    else:
        if beat_map and beat_map.get("chunks"):
            user_log("Story beats resume", f"{int(beat_map.get('completed_chunks', 0) or 0)} chunks already saved")
        beat_map = build_beat_map(
            transcript,
            call_beat_llm,
            analysis_map=analysis_map,
            episode_digest=episode_digest,
            cache_path=beat_map_json,
        )
        beat_map = _snap_beat_boundaries_to_utterances(beat_map, analysis_map)
        write_json(beat_map_json, beat_map)
        user_log("Story beats ready", f"{len(beat_map.get('beats', []))} complete beats found")

    stage("GPT clip selection", "ranking story beats by hook, payoff, audio energy, and scene flow")
    highlights_result = read_json(highlights_json)
    if highlights_result and highlights_result.get("complete"):
        user_log("Candidates ready", f"{len(highlights_result.get('highlights', []))} candidates loaded from cache")
    else:
        if highlights_result and highlights_result.get("batches"):
            user_log("Candidates resume", f"{int(highlights_result.get('completed_batches', 0) or 0)} batches already saved")
        elif highlights_result:
            user_log("Candidates cache incomplete", f"rebuilding up to {num_clips} shorts")
        highlights_result = get_highlights(
            transcript,
            num_clips=num_clips,
            llm_fn=call_beat_llm,
            review_llm_fn=call_fast_llm,
            beat_map=beat_map,
            analysis_map=analysis_map,
            episode_digest=episode_digest,
            cache_path=highlights_json,
        )
        highlights_result["highlights"] = keep_postable_highlights(
            score_highlights(highlights_result.get("highlights", []), analysis_map)
        )
        write_json(highlights_json, highlights_result)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        user_log("No strong clips", "Candidate review rejected every moment as not postable enough")
        result = {
            "source_video_url": source_path,
            "transcript": transcript,
            "highlights": [],
            "shorts": [],
        }
        write_json(result_json, result)
        return result

    verified_top_state = read_json(verified_top_json)
    verified_top = verified_top_state.get("highlights") if verified_top_state else None
    if verified_top and (
        len(verified_top) >= num_clips or verified_top_state.get("quality_limited") is True
    ) and verified_top_state.get("boundary_review_version") == BOUNDARY_REVIEW_VERSION and verified_top_state.get("final_quality_version") == FINAL_QUALITY_VERSION:
        top = verified_top[:num_clips]
        user_log("Final picks ready", f"{len(top)} clips loaded from cache")
    else:
        top_state = read_json(top_json)
        cached_top = top_state.get("highlights") if top_state else None
        if cached_top:
            user_log("Checking cached picks", f"{len(cached_top)} clips")
            top = cached_top[:num_clips]
        else:
            top = score_highlights(
                sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True),
                analysis_map,
            )[:num_clips]
            write_json(top_json, {"highlights": top})
        stage("Checking clip boundaries", f"LLM adjusts {len(top)} clips to semantic scene endings")
        top = refine_highlight_boundaries_with_llm(transcript, top, call_fast_llm, episode_digest=episode_digest)
        top = _apply_text_anchor_boundaries(top, transcript)
        top = _snap_highlights_to_segments(top, transcript)
        top = keep_postable_highlights(_enforce_duration_cap(top, transcript))
        stage("Final quality gate", "checking hooks, duplicates, upload metadata, and render style")
        top = final_quality_review_with_llm(
            transcript,
            top,
            num_clips,
            call_fast_llm,
            episode_digest=episode_digest,
            analysis_map=analysis_map,
            cache_path=final_quality_json,
        )
        top = _snap_highlights_to_segments(top, transcript)
        top = keep_postable_highlights(_enforce_duration_cap(top, transcript), min_score=82)
        write_json(verified_top_json, {
            "highlights": top,
            "quality_limited": len(top) < num_clips,
            "boundary_review_version": BOUNDARY_REVIEW_VERSION,
            "final_quality_version": FINAL_QUALITY_VERSION,
        })
    if not top:
        user_log("No strong clips", "Final quality check rejected every candidate")
        result = {
            "source_video_url": source_path,
            "transcript": transcript,
            "highlights": all_highlights,
            "shorts": [],
        }
        write_json(result_json, result)
        return result
    stage("Rendering shorts", f"{len(top)} final clips selected from {len(all_highlights)} candidates")
    old_edit_env = {
        "LOCAL_TIGHTEN_PAUSES": os.environ.get("LOCAL_TIGHTEN_PAUSES"),
        "LOCAL_PAUSE_THRESHOLD": os.environ.get("LOCAL_PAUSE_THRESHOLD"),
        "LOCAL_PAUSE_KEEP": os.environ.get("LOCAL_PAUSE_KEEP"),
    }
    os.environ["LOCAL_TIGHTEN_PAUSES"] = "1" if edit_plan.get("tighten_pauses") else "0"
    os.environ["LOCAL_PAUSE_THRESHOLD"] = str(edit_plan.get("pause_threshold", 0.75))
    os.environ["LOCAL_PAUSE_KEEP"] = str(edit_plan.get("pause_keep", 0.24))

    try:
        shorts = crop_highlights_local(
            source_path,
            top,
            aspect_ratio=aspect_ratio,
            out_dir=output_dir,
            transcript=transcript,
        )
    finally:
        for key, value in old_edit_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for index, short in enumerate(shorts, 1):
        clip_url = short.get("clip_url")
        if clip_url and os.path.exists(clip_url):
            public_path = os.path.join(output_dir, f"short_{index:02d}.mp4")
            if os.path.abspath(clip_url) != os.path.abspath(public_path):
                shutil.copy2(clip_url, public_path)
            short["clip_url"] = public_path
    metadata_path = _write_upload_metadata(output_dir, shorts)
    contact_sheet_path = _write_contact_sheet(output_dir, shorts)
    user_log("Upload kit", f"metadata and review sheet saved: {os.path.basename(metadata_path)}, {os.path.basename(contact_sheet_path)}")
    result = {
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
        "upload_metadata": metadata_path,
        "contact_sheet": contact_sheet_path,
    }
    write_json(result_json, result)

    return result
