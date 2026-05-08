"""End-to-end orchestrator — local mode only.

Pipeline: yt-dlp download -> faster-whisper transcribe -> OpenAI LLM highlights -> ffmpeg/opencv render.
All processing runs locally. OPENAI_API_KEY required for highlight analysis.
"""
import os
import shutil
from typing import Dict, List, Optional

from .config import edit_profile, local_output_dir
from .highlights import build_beat_map, decide_auto_clip_count, decide_edit_plan, get_highlights, keep_postable_highlights, verify_highlights_with_llm


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


MAX_CLIP_SECONDS = 60  # hard cap — no clip exceeds 60s


def _has_natural_pause_after(segments: List[Dict], index: int, duration: float) -> bool:
    segment = segments[index]
    seg_end = float(segment["end"])
    next_start = float(segments[index + 1]["start"]) if index + 1 < len(segments) else duration
    return next_start - seg_end >= 0.55


def _ends_like_sentence(text: str) -> bool:
    return (text or "").strip().endswith((".", "!", "?", "...", "…", "вЂ¦"))


def _natural_end_before_cap(start: float, end: float, transcript: Dict) -> float:
    segments = transcript.get("segments", [])
    if end - start <= MAX_CLIP_SECONDS or not segments:
        return end

    max_end = start + MAX_CLIP_SECONDS
    duration = float(transcript.get("duration", 0.0) or 0.0)
    eligible = [
        (index, segment)
        for index, segment in enumerate(segments)
        if float(segment["end"]) <= max_end and float(segment["end"]) > start + 10.0
    ]
    if not eligible:
        return max_end

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

    duration = float(transcript.get("duration", 0.0) or 0.0)
    fixed = []
    for highlight in highlights:
        start = max(0.0, float(highlight.get("start_time", 0.0)))
        end = float(highlight.get("end_time", 0.0))
        max_end = start + MAX_CLIP_SECONDS

        containing = _segment_index_at(segments, end)
        if containing is not None:
            seg_end = float(segments[containing]["end"])
            if seg_end <= max_end:
                end = seg_end + 0.25
            else:
                previous = [
                    s for s in segments
                    if float(s["end"]) <= max_end and float(s["end"]) > start + 10.0
                ]
                if previous:
                    end = float(previous[-1]["end"]) + 0.25
                else:
                    end = max_end

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

        if duration > 0:
            end = min(end, duration)
        end = min(end, max_end)
        if end - start < 10.0:
            end = min(max_end, start + 10.0, duration or start + 10.0)
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
    """Hard-cap every clip to MAX_CLIP_SECONDS, trimming the end."""
    capped = []
    for h in highlights:
        start = float(h.get("start_time", 0))
        end = float(h.get("end_time", 0))
        if end - start > MAX_CLIP_SECONDS:
            end = _natural_end_before_cap(start, end, transcript)
            h = {**h, "end_time": end}
        capped.append(h)
    return _complete_clip_boundaries(capped, transcript)


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
    from .local.llm import call_openai_llm
    from .local.media_analysis import build_analysis_map, score_highlights
    from .local.progress import stage, user_log
    from .local.session import hydrate_from_matching_source, hydrate_latest_transcript, read_json, session_dir, write_json
    from .local.transcriber import transcribe_local

    output_dir = local_output_dir()
    session_path = session_dir(video_url, download_format, language)
    source_json = f"{session_path}/source.json"
    transcript_json = f"{session_path}/transcript.json"
    analysis_map_json = f"{session_path}/analysis_map.json"
    beat_map_json = f"{session_path}/beat_map.json"
    highlights_json = f"{session_path}/highlights.json"
    auto_plan_json = f"{session_path}/auto_plan.json"
    edit_plan_json = f"{session_path}/edit_plan.json"
    top_json = f"{session_path}/top.json"
    verified_top_json = f"{session_path}/verified_top.json"
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
    if analysis_map and analysis_map.get("utterances"):
        user_log(
            "Media analysis ready",
            f"{len(analysis_map.get('utterances', []))} dialogue blocks, "
            f"{max(len(analysis_map.get('scene_cuts', [])) - 1, 0)} visual cuts",
        )
    else:
        analysis_map = build_analysis_map(source_path, transcript)
        write_json(analysis_map_json, analysis_map)

    requested_edit_profile = edit_profile()
    stage("Choosing edit style", f"profile setting: {requested_edit_profile}")
    edit_plan = read_json(edit_plan_json)
    if not edit_plan or edit_plan.get("requested_profile") != requested_edit_profile:
        edit_plan = decide_edit_plan(transcript, call_openai_llm, requested_profile=requested_edit_profile)
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
            auto_plan = decide_auto_clip_count(transcript, call_openai_llm)
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

    stage("Finding story beats", "grouping the episode into complete jokes/scenes/ideas")
    beat_map = read_json(beat_map_json)
    if beat_map and beat_map.get("beats"):
        user_log("Story beats ready", f"{len(beat_map.get('beats', []))} beats loaded from cache")
    else:
        beat_map = build_beat_map(transcript, call_openai_llm, analysis_map=analysis_map)
        beat_map = _snap_beat_boundaries_to_utterances(beat_map, analysis_map)
        write_json(beat_map_json, beat_map)
        user_log("Story beats ready", f"{len(beat_map.get('beats', []))} complete beats found")

    stage("Choosing candidate shorts", "ranking beats by hook, payoff, audio energy, and scene flow")
    highlights_result = read_json(highlights_json)
    if highlights_result and len(highlights_result.get("highlights", [])) >= num_clips:
        user_log("Candidates ready", f"{len(highlights_result.get('highlights', []))} candidates loaded from cache")
    else:
        if highlights_result:
            user_log("Candidates cache incomplete", f"rebuilding {num_clips} shorts")
        highlights_result = get_highlights(
            transcript,
            num_clips=num_clips,
            llm_fn=call_openai_llm,
            beat_map=beat_map,
            analysis_map=analysis_map,
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
    ):
        top = verified_top[:num_clips]
        user_log("Final picks ready", f"{len(top)} clips loaded from cache")
    else:
        top_state = read_json(top_json)
        cached_top = top_state.get("highlights") if top_state else None
        if cached_top:
            user_log("Checking cached picks", f"{len(cached_top)} clips")
            verifier_candidates = cached_top
        else:
            verifier_candidates = score_highlights(
                sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True),
                analysis_map,
            )
            write_json(top_json, {"highlights": verifier_candidates})
        stage("Checking clip boundaries", f"making sure {num_clips} clips end on complete thoughts")
        top = verify_highlights_with_llm(transcript, verifier_candidates, num_clips, call_openai_llm)
        top = _snap_highlights_to_segments(top, transcript)
        top = keep_postable_highlights(_enforce_duration_cap(top, transcript))
        write_json(verified_top_json, {"highlights": top, "quality_limited": len(top) < num_clips})
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
    result = {
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }
    write_json(result_json, result)

    return result
