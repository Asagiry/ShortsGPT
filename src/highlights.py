"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive any OpenAI-compatible API.
"""
import json
import re
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from .local.progress import Progress, user_log
from .local.session import read_json, write_json


LLMFn = Callable[[str], str]

MIN_POSTABLE_SCORE = 78
HIGHLIGHT_SELECTION_VERSION = 4
FINAL_QUALITY_REVIEW_VERSION = 3


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


AUTO_CLIP_COUNT_PROMPT = """You are planning a batch of short-form clips from a video transcript.

Choose the maximum number of shorts that are worth generating naturally. Do NOT choose based on video duration.
Choose based on how many distinct, self-contained, viral-worthy ideas, stories, jokes, arguments, revelations, or emotional beats are actually present.

Rules:
- Count only moments that can stand alone as a short. Aim for 10-60 seconds, but allow 61-75 seconds only when needed to keep the final phrase/reaction complete.
- Multiple clips are good only when they cover different meanings or beats.
- Do not force filler clips just to make more output.
- It is better to output 2 great clips than 12 average clips.
- If the video has no genuinely postable moments, choose 0.
- If the transcript has one strong central idea, choose 1.
- If it has several separate strong ideas, choose the number of those ideas.
- Maximum 16 clips.

Respond with JSON only:
{"num_clips": int, "reason": "short explanation"}"""


EDIT_PLAN_PROMPT = """You are choosing editing rhythm for automatic short-form clipping.

Choose an edit profile and pause-removal settings for this specific video.

Profiles:
- talking_head: podcast, interview, lecture, commentary, streamer talking to camera. Tight cuts are good.
- cartoon_dialogue: animated sitcoms/cartoons/series dialogue. Preserve reaction beats and joke timing.
- movie_scene: film scenes, drama, cinematic scenes. Preserve pauses, mood, tension, music, visual acting.
- gameplay: gameplay/screen content where visual events matter more than speech pauses.
- music_visual: music, performance, montage, trailers. Do not cut by speech pauses aggressively.

Return:
- profile: one of talking_head, cartoon_dialogue, movie_scene, gameplay, music_visual
- tighten_pauses: true/false
- pause_threshold: seconds of silence/gap before cutting. Range 0.6-2.5
- pause_keep: seconds of pause to keep around a cut. Range 0.15-0.9
- reason: short explanation

Guidance:
- talking_head: threshold 0.6-0.9, keep 0.18-0.35
- cartoon_dialogue: threshold 1.6-2.2, keep 0.65-0.9. Never remove silent reaction shots or visual punchline landings.
- movie_scene: usually tighten_pauses false, or threshold 1.8-2.5, keep 0.7-0.9
- gameplay: threshold 1.4-2.2, keep 0.45-0.8
- music_visual: tighten_pauses false

Respond JSON only:
{"profile":"...","tighten_pauses":true,"pause_threshold":1.2,"pause_keep":0.5,"reason":"..."}"""


EPISODE_DIGEST_PROMPT = """You are preparing an editorial brief for clipping a scripted episode into short-form videos.

Analyze the transcript like a story editor, not like a keyword extractor.

Return:
- logline: one sentence summary of the episode
- main_conflict: the central story/problem
- characters: important characters and what they want
- recurring_jokes: repeated jokes/running bits if present
- story_turns: major plot turns in order
- clip_strategy: what kind of moments should be clipped for this episode
- avoid: what would make bad shorts for this episode

Rules:
- Keep it compact and practical for another LLM choosing clips.
- Do not invent facts not present in transcript.
- For cartoons/sitcoms, emphasize setup -> reversal -> punchline/reaction.

Respond JSON only:
{"logline":"...","main_conflict":"...","characters":[{"name":"...","role":"...","wants":"..."}],"recurring_jokes":["..."],"story_turns":["..."],"clip_strategy":["..."],"avoid":["..."]}"""


BEAT_MAP_PROMPT = """You are making an editorial beat map from a timestamped transcript and local media analysis.

Do not select final clips yet. First divide the transcript into complete meaning units.
A beat is a self-contained story moment, joke setup/payoff, argument, reveal, conflict, reaction, lesson, or scene turn.
For scripted shows, read it like an editor who understands the episode: character goal, conflict, reversal, joke setup, payoff, and what the scene means in the story.

For each beat:
- start_time: where necessary setup begins
- end_time: where the payoff/conclusion/reaction lands
- type: joke, conflict, reveal, lesson, emotional_turn, useful_info, scene, other
- summary: what happens in plain language
- setup: what context the viewer needs
- payoff: how the beat resolves
- standalone: whether it can work as a short without outside context
- visual_dependency: low, medium, high
- clip_potential: 0-100 for short-form value
- boundary_note: what must not be cut off at the end

Rules:
- Prefer complete beats over tiny transcript fragments.
- Preserve the scene's purpose. A clip should not be just a funny line if the viewer needs the setup or the aftermath to understand why it matters.
- For comedy/cartoon/dialogue, include the reaction after the punchline when it matters.
- If the transcript goes quiet after a line but the beat likely continues visually, include that visual/reaction tail in end_time.
- A beat can be longer than 60 seconds; final clipping will choose sub-beats later.
- Use utterance IDs when possible, because final cuts map utterance boundaries more reliably than raw timestamps.
- Use scene cuts and audio peaks as supporting evidence. Strong clips usually have coherent visual scenes and/or high audio momentum.
- Do not invent events not present in the transcript.

Respond JSON only:
{"beats":[{"start_time":float,"end_time":float,"utterance_start_id":"u0001","utterance_end_id":"u0003","type":"...","summary":"...","setup":"...","payoff":"...","standalone":true,"visual_dependency":"medium","clip_potential":85,"boundary_note":"..."}]}"""


VIRALITY_CRITERIA = """
Editorial goal: find complete short-form moments, not chopped transcript fragments.

What makes a good clip:
- A clear opening hook: the first sentence should make sense and create curiosity.
- One complete beat: one idea, conflict, joke, reveal, decision, lesson, or emotional turn.
- A natural ending: finish after the payoff, conclusion, answer, reaction, or final sentence of the beat.
- For scripted scenes/cartoons, a silent face reaction, object reveal, cutaway, or visual consequence can be the ending. Do not trim it just because nobody is speaking.
- Self-contained context: a viewer who sees only this clip should understand why it matters.
- Strong retention: remove obvious warm-up, filler, long pauses, and repeated phrases when they are not needed.
- It should feel worth posting. If it is merely understandable but not funny, surprising, emotional, useful, tense, or shareable, reject it.

Duration guidance:
- Prefer 18-45 seconds when that preserves the full idea.
- 10-17 seconds is fine for a very complete punchline or one-liner.
- Aim for 60 seconds or less. If the complete ending lands slightly later, 61-75 seconds is acceptable. If a beat needs more than 75 seconds, choose a tighter complete sub-beat or skip it.
- Do not cut the ending just to make the clip shorter. Completeness beats aggressive trimming.

Good signals:
- A surprising claim, turn, confession, disagreement, lesson, strong opinion, emotional reaction, joke payoff, or concrete useful advice.
- A moment where the speaker's thought has a beginning, development, and landing.

Reject:
- Average moments that only exist to fill a requested count.
- Mild dialogue with no strong turn, punchline, reveal, conflict, reaction, or useful takeaway.
- A setup without the payoff.
- A payoff without enough setup to understand it.
- A clip that ends mid-thought, mid-answer, before the reaction/conclusion, or before the visual gag lands.
- Multiple unrelated ideas forced into one clip.
- Overlapping clips that say the same thing.
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are a senior short-form editor selecting finished moments from a timestamped transcript.

{virality_criteria}

Content type: {content_type} | Density: {density}

You are given a timestamped transcript. Your job is to find the best complete short clips.

Timestamp rules:
- start_time should be the first sentence/phrase needed for the viewer to understand the hook.
- end_time should be after the thought lands: conclusion, payoff, answer, reaction, or final sentence.
- If the selected moment ends with a visual/silent reaction, include that reaction tail.
- Use transcript timestamps closely, but choose the range that preserves the complete meaning.
- Do not end before the speaker finishes the selected idea.
- Prefer ending at the end of a transcript segment or after a clear pause. Never place end_time in the middle of an ongoing phrase.
- Do not add unrelated context before or after the selected beat.
- Clip duration should usually be 10-60 seconds. Use 61-75 seconds only to preserve a complete final sentence, answer, punchline, or reaction. If the complete idea would exceed 75 seconds, select a smaller complete sub-beat instead of truncating it.

MANDATORY RULES:
1. COMPLETE MEANING - never cut off the ending of the selected idea.
2. ONE BEAT - each clip should cover one coherent moment.
3. CLEAN EDGES - remove unrelated intro/outro, but keep necessary setup and payoff.
4. HOOK FIRST - identify the strongest opening sentence as hook_sentence.
5. NO OVERLAP - clips must not overlap.
6. SCORE 0-100 - score for retention, clarity, emotional pull, and completeness.
7. {num_clips_instruction}
8. For each highlight, explain why this clip works as a short ("virality_reason").
9. Include score_matrix: hook, standalone_clarity, setup_completeness, payoff_strength, ending_completeness, shareability, context_loss_risk.

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string","score_matrix":{{"hook":0,"standalone_clarity":0,"setup_completeness":0,"payoff_strength":0,"ending_completeness":0,"shareability":0,"context_loss_risk":0}}}}]}}"""


BEAT_HIGHLIGHT_PROMPT = """You are a senior short-form editor selecting final shorts from an editorial beat map.

{virality_criteria}

Content type: {content_type} | Density: {density}

Episode editorial brief:
{episode_context}

Choose the best complete clips from the beats below.

Rules:
- Pick clips from beats with high clip_potential and standalone=true when possible.
- Return fewer clips than requested when the batch is weak. Returning zero is correct for a weak batch.
- Do not include average/filler moments just to increase output count.
- Preserve setup -> development -> payoff/reaction.
- For scripted episodes, prefer moments that carry a complete mini-story: setup, character intention, conflict/reversal, and landing.
- Do not pick isolated punchlines, random exposition, or half-scenes just because the line is funny.
- If a beat is longer than 75 seconds, choose a smaller complete sub-beat inside it.
- Never end in the middle of a sentence, answer, joke, reaction, or boundary_note.
- Never shorten a clip by cutting off the last reaction, rebuttal, object reveal, or visual landing.
- Prefer boundaries aligned to utterance_start_id/utterance_end_id and nearby scene cuts.
- Favor candidates with strong local_score/audio_peak_ratio unless visual_dependency is high and the transcript alone is insufficient.
- For cartoons/comedy/dialogue, do not cut before the reaction or landing if it is part of the joke.
- Clip duration should usually be 10-60 seconds; 61-75 seconds is allowed only to avoid cutting the ending.
- Include score_matrix: hook, standalone_clarity, setup_completeness, payoff_strength, ending_completeness, shareability, context_loss_risk.
- {num_clips_instruction}

Respond ONLY with valid JSON:
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string","score_matrix":{{"hook":0,"standalone_clarity":0,"setup_completeness":0,"payoff_strength":0,"ending_completeness":0,"shareability":0,"context_loss_risk":0}}}}]}}

Beat map:
{beat_map_json}"""


FINAL_RERANK_PROMPT = """You are the final shorts editor.

Choose the best {num_clips} clips from the candidate list.
Treat {num_clips} as a hard maximum, not a target. Keep only clips that are complete, standalone, and genuinely worth posting.

Rules:
- Select 0-{num_clips} clips. Fewer is better than filler.
- Reject average clips even if that means the final output has only a few shorts.
- Keep only clips with a strong hook and clear payoff/reaction/reveal/conflict/useful takeaway.
- Prefer complete setup -> payoff/reaction clips.
- Do not choose overlapping clips.
- Do not choose clips ending mid-sentence or before the reaction/landing.
- If preserving the final reaction requires trimming, move start_time later rather than cutting end_time earlier.
- Keep clips around 10-60 seconds; 61-75 seconds is allowed only when needed for a complete ending.
- You may adjust start_time/end_time slightly to clean boundaries.

Respond JSON only:
{{"selected":[{{"id":int,"start_time":float,"end_time":float,"score":int,"reason":"..."}}]}}

Candidates:
{candidates_json}"""


FINAL_QUALITY_PROMPT = """You are the release editor for a short-form channel.

Your job is to decide what is actually worth rendering and uploading.
Do not protect previous choices. Reject anything that is average, unclear, duplicated, or incomplete.

Editorial brief:
{episode_context}

Content type: {content_type} | Density: {density}

Platform goals:
- Strong first 1-3 seconds.
- A complete mini-story, joke, conflict, reveal, lesson, or emotional turn.
- Clear payoff/reaction/conclusion.
- Viewer understands the moment without seeing the full episode.
- No duplicate ideas in the final set.

Content-specific rules:
- cartoon_dialogue/sitcom: setup -> reversal -> punchline/reaction. Keep reaction timing.
- movie_scene: conflict/tension -> turn -> emotional or narrative landing. Do not over-tighten pauses.
- talking_head/podcast/interview: claim/story -> proof/detail -> takeaway. Tighten filler.
- gameplay/screen: visible event -> reaction/result. Do not cut away from the visual event.

For each kept clip:
- You may adjust start_time/end_time only to improve hook, context, and complete ending.
- Create a short intro_overlay only if the first seconds need context. Max 55 characters. Empty string if not needed.
- Provide highlight_keywords for captions: 2-6 words/phrases that should pop visually.
- Provide pause_policy: keep_reactions, balanced, or tight.
- Provide upload metadata: 3 titles, description, hashtags, pinned_comment.
- Provide score_matrix and viral_score 0-100.
- Provide semantic_key, a short meaning label used to remove duplicate ideas.

Hard quality gate:
- Keep clips with viral_score >= 70.
- Reject clips with weak first 3 seconds, missing setup, missing payoff, unclear context, duplicated meaning, or ending before the thought lands.
- If only 2 clips are great, keep 2. If none are great, keep none.

Respond JSON only:
{{"clips":[{{"id":int,"keep":true,"start_time":float,"end_time":float,"viral_score":int,"score_matrix":{{"hook":0,"first_3s":0,"standalone_clarity":0,"setup_completeness":0,"payoff_strength":0,"ending_completeness":0,"shareability":0,"rewatch_potential":0,"context_loss_risk":0}},"semantic_key":"...","intro_overlay":"...","hook_sentence":"...","pause_policy":"balanced","highlight_keywords":["..."],"titles":["...","...","..."],"description":"...","hashtags":["#..."],"pinned_comment":"...","reason":"..."}}]}}

Candidates:
{candidates_json}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
BEAT_MAP_CHUNK_SECONDS = 300
BEAT_MAP_OVERLAP_SECONDS = 45
BEAT_SELECT_BATCH_SIZE = 12


def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"LLM returned incomplete JSON ({len(raw)} chars): {e}") from e


def detect_content_type(transcript: Dict, llm_fn: LLMFn = None) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def _segments_between(transcript: Dict, start: float, end: float) -> List[Dict]:
    return [
        s for s in transcript.get("segments", [])
        if float(s.get("end", 0.0)) >= start and float(s.get("start", 0.0)) <= end
    ]


def _transcript_text_for_window(transcript: Dict, start: float, end: float) -> str:
    return "\n".join(
        f"[{float(s['start']):.1f}s] {str(s.get('text', '')).strip()}"
        for s in _segments_between(transcript, start, end)
        if str(s.get("text", "")).strip()
    )


def _analysis_for_window(analysis_map: Optional[Dict], start: float, end: float) -> Dict:
    if not analysis_map:
        return {}
    utterances = [
        u for u in analysis_map.get("utterances", [])
        if float(u.get("end", 0.0)) >= start and float(u.get("start", 0.0)) <= end
    ]
    scene_cuts = [
        c for c in analysis_map.get("scene_cuts", [])
        if start <= float(c.get("time", 0.0)) <= end
    ]
    scene_windows = [
        s for s in analysis_map.get("scene_windows", [])
        if float(s.get("end", 0.0)) >= start and float(s.get("start", 0.0)) <= end
    ]
    audio = [
        w for w in analysis_map.get("audio_energy", [])
        if w.get("peak") and float(w.get("end", 0.0)) >= start and float(w.get("start", 0.0)) <= end
    ]
    return {
        "utterances": [
            {**u, "text": _compact_text(str(u.get("text", "")), 180)}
            for u in utterances[:80]
        ],
        "scene_cuts": scene_cuts[:80],
        "scene_windows": scene_windows[:40],
        "audio_peak_windows": audio[:60],
    }


def _compact_text(value: str, limit: int = 260) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _episode_context_text(episode_digest: Optional[Dict]) -> str:
    if not episode_digest:
        return "No episode brief available."
    return json.dumps(episode_digest, ensure_ascii=False, separators=(",", ":"))[:5000]


def _dedupe_beats(beats: List[Dict]) -> List[Dict]:
    beats = sorted(beats, key=lambda b: int(b.get("clip_potential", 0) or 0), reverse=True)
    kept: List[Dict] = []
    for beat in beats:
        start = float(beat.get("start_time", 0.0))
        end = float(beat.get("end_time", 0.0))
        duration = max(0.1, end - start)
        duplicate = False
        for existing in kept:
            latest_start = max(start, float(existing.get("start_time", 0.0)))
            earliest_end = min(end, float(existing.get("end_time", 0.0)))
            overlap = max(0.0, earliest_end - latest_start)
            if overlap / duration > 0.65:
                duplicate = True
                break
        if not duplicate:
            kept.append(beat)
    return sorted(kept, key=lambda b: float(b.get("start_time", 0.0)))


def _parse_beats(data: Dict) -> List[Dict]:
    beats = []
    for beat in data.get("beats", []):
        try:
            start = float(beat.get("start_time", 0.0))
            end = float(beat.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        beats.append({
            "start_time": start,
            "end_time": end,
            "utterance_start_id": str(beat.get("utterance_start_id", "")),
            "utterance_end_id": str(beat.get("utterance_end_id", "")),
            "type": str(beat.get("type", "other")),
            "summary": str(beat.get("summary", "")),
            "setup": str(beat.get("setup", "")),
            "payoff": str(beat.get("payoff", "")),
            "standalone": bool(beat.get("standalone", False)),
            "visual_dependency": str(beat.get("visual_dependency", "medium")),
            "clip_potential": int(beat.get("clip_potential", 0) or 0),
            "boundary_note": str(beat.get("boundary_note", "")),
        })
    return beats


def _fallback_highlights_from_beats(beats: List[Dict], num_clips: int, reason: str = "LLM returned zero candidates") -> List[Dict]:
    fallback: List[Dict] = []
    ranked = sorted(
        beats,
        key=lambda b: (
            int(b.get("clip_potential", 0) or 0),
            1 if b.get("standalone") else 0,
            float(b.get("end_time", 0.0)) - float(b.get("start_time", 0.0)),
        ),
        reverse=True,
    )
    for beat in ranked:
        try:
            start = float(beat.get("start_time", 0.0))
            end = float(beat.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if end - start < 10.0:
            end = start + 10.0
        if end - start > 75.0:
            end = start + 75.0
        potential = int(beat.get("clip_potential", 0) or 0)
        score = max(68, min(88, potential if potential else 72))
        title = _compact_text(str(beat.get("summary") or beat.get("payoff") or "Episode moment"), 70)
        fallback.append({
            "title": title,
            "start_time": start,
            "end_time": end,
            "score": score,
            "viral_score": score,
            "hook_sentence": _compact_text(str(beat.get("setup") or title), 120),
            "virality_reason": f"{reason}; kept from story beat: {_compact_text(str(beat.get('payoff') or beat.get('summary') or ''), 180)}",
            "semantic_key": _compact_text(str(beat.get("summary") or title).casefold(), 90),
            "score_matrix": {
                "hook": score,
                "first_3s": max(60, score - 5),
                "standalone_clarity": 70 if beat.get("standalone") else 60,
                "setup_completeness": 70,
                "payoff_strength": max(62, score - 4),
                "ending_completeness": 70,
                "shareability": max(58, score - 8),
                "rewatch_potential": 58,
                "context_loss_risk": 35 if beat.get("standalone") else 55,
            },
        })
        if len(fallback) >= max(1, min(num_clips, 4)):
            break
    return fallback


def build_transcript_sample(transcript: Dict, max_chars: int = 14000) -> str:
    text = build_transcript_text(transcript)
    if len(text) <= max_chars:
        return text

    head_size = max_chars // 3
    tail_size = max_chars // 3
    middle_size = max_chars - head_size - tail_size - 80
    middle_start = max(0, (len(text) - middle_size) // 2)
    return (
        text[:head_size]
        + "\n\n[...middle sample...]\n\n"
        + text[middle_start:middle_start + middle_size]
        + "\n\n[...ending sample...]\n\n"
        + text[-tail_size:]
    )


def decide_auto_clip_count(
    transcript: Dict,
    llm_fn: LLMFn,
    max_clips: int = 16,
) -> Dict:
    duration = float(transcript.get("duration", 0.0) or 0.0)
    sample = build_transcript_sample(transcript)
    prompt = (
        f"{AUTO_CLIP_COUNT_PROMPT}\n\n"
        f"Video duration: {duration:.0f}s\n\n"
        f"Transcript sample:\n{sample}"
    )
    raw = llm_fn(prompt)
    data = _parse_json_loose(raw)
    count = int(data.get("num_clips", 0))
    if count < 0:
        raise ValueError("AUTO clip count failed: LLM returned negative num_clips")
    count = min(max(count, 0), max_clips)
    return {
        "num_clips": count,
        "reason": str(data.get("reason") or "GPT selected a natural clip count."),
        "source": "llm",
    }


def decide_edit_plan(
    transcript: Dict,
    llm_fn: LLMFn,
    requested_profile: str = "auto",
) -> Dict:
    requested_profile = (requested_profile or "auto").strip().lower()
    defaults = {
        "talking_head": {"tighten_pauses": True, "pause_threshold": 0.75, "pause_keep": 0.24},
        "cartoon_dialogue": {"tighten_pauses": True, "pause_threshold": 1.4, "pause_keep": 0.55},
        "movie_scene": {"tighten_pauses": False, "pause_threshold": 2.0, "pause_keep": 0.8},
        "gameplay": {"tighten_pauses": True, "pause_threshold": 1.7, "pause_keep": 0.6},
        "music_visual": {"tighten_pauses": False, "pause_threshold": 2.2, "pause_keep": 0.8},
    }
    if requested_profile in defaults:
        return {
            "profile": requested_profile,
            **defaults[requested_profile],
            "source": "manual",
            "reason": "Manual edit profile selected.",
        }

    sample = build_transcript_sample(transcript, max_chars=10000)
    prompt = f"{EDIT_PLAN_PROMPT}\n\nTranscript sample:\n{sample}"
    raw = llm_fn(prompt)
    data = _parse_json_loose(raw)
    profile = str(data.get("profile", "talking_head")).strip().lower()
    if profile not in defaults:
        raise ValueError(f"Edit plan failed: unknown LLM profile {profile!r}")
    base = dict(defaults[profile])
    base.update({
        "profile": profile,
        "tighten_pauses": bool(data.get("tighten_pauses", base["tighten_pauses"])),
        "pause_threshold": min(max(float(data.get("pause_threshold", base["pause_threshold"])), 0.6), 2.5),
        "pause_keep": min(max(float(data.get("pause_keep", base["pause_keep"])), 0.15), 0.9),
        "reason": str(data.get("reason") or "LLM selected edit rhythm."),
        "source": "llm",
    })
    return base


def build_episode_digest(
    transcript: Dict,
    llm_fn: LLMFn,
    analysis_map: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> Dict:
    cached = read_json(cache_path) if cache_path else None
    if cached and cached.get("complete"):
        return cached

    sample = build_transcript_sample(transcript, max_chars=18000)
    analysis_json = json.dumps(
        _compact_analysis_for_prompt(analysis_map),
        ensure_ascii=False,
        separators=(",", ":"),
    )[:7000]
    prompt = (
        f"{EPISODE_DIGEST_PROMPT}\n\n"
        f"Local media analysis summary:\n{analysis_json}\n\n"
        f"Timestamped transcript sample:\n{sample}"
    )
    user_log("Episode brief", "understanding story, characters, and clip strategy")
    data = _parse_json_loose(llm_fn(prompt))
    digest = {
        "complete": True,
        "logline": str(data.get("logline", "")),
        "main_conflict": str(data.get("main_conflict", "")),
        "characters": data.get("characters", []) if isinstance(data.get("characters", []), list) else [],
        "recurring_jokes": data.get("recurring_jokes", []) if isinstance(data.get("recurring_jokes", []), list) else [],
        "story_turns": data.get("story_turns", []) if isinstance(data.get("story_turns", []), list) else [],
        "clip_strategy": data.get("clip_strategy", []) if isinstance(data.get("clip_strategy", []), list) else [],
        "avoid": data.get("avoid", []) if isinstance(data.get("avoid", []), list) else [],
    }
    if cache_path:
        write_json(cache_path, digest)
    return digest


def _compact_analysis_for_prompt(analysis_map: Optional[Dict]) -> Dict:
    if not analysis_map:
        return {}
    utterances = analysis_map.get("utterances", [])
    scene_cuts = analysis_map.get("scene_cuts", [])
    scene_windows = analysis_map.get("scene_windows", [])
    audio = analysis_map.get("audio_energy", [])
    peak_windows = [w for w in audio if w.get("peak")]
    return {
        "utterances": [
            {**u, "text": _compact_text(str(u.get("text", "")), 180)}
            for u in utterances[:180]
        ],
        "scene_cuts": scene_cuts[:180],
        "scene_windows": scene_windows[:80],
        "audio_peak_windows": peak_windows[:140],
    }


def _beat_map_chunks(transcript: Dict) -> List[Dict]:
    duration = float(transcript.get("duration", 0.0) or 0.0)
    step = BEAT_MAP_CHUNK_SECONDS - BEAT_MAP_OVERLAP_SECONDS
    chunks = []
    start = 0.0
    index = 1
    while start < max(duration, 1.0):
        end = min(duration, start + BEAT_MAP_CHUNK_SECONDS)
        chunks.append({"index": index, "start": start, "end": end})
        if end >= duration:
            break
        start += step
        index += 1
    return chunks or [{"index": 1, "start": 0.0, "end": duration}]


def _partial_chunks(existing: Optional[Dict]) -> Dict[int, Dict]:
    chunks = existing.get("chunks", []) if existing else []
    return {int(item.get("index", 0)): item for item in chunks if int(item.get("index", 0) or 0) > 0}


def _partial_batches(existing: Optional[Dict]) -> Dict[int, Dict]:
    batches = existing.get("batches", []) if existing else []
    return {int(item.get("index", 0)): item for item in batches if int(item.get("index", 0) or 0) > 0}


def _llm_parallelism(total_items: int) -> int:
    try:
        configured = int(os.getenv("LLM_PARALLELISM", "2"))
    except ValueError:
        configured = 2
    return max(1, min(total_items, configured, 4))


def _write_partial_beat_map(cache_path: Optional[str], chunks: List[Dict], complete: bool = False) -> None:
    if not cache_path:
        return
    all_beats: List[Dict] = []
    for chunk in sorted(chunks, key=lambda item: int(item.get("index", 0) or 0)):
        all_beats.extend(chunk.get("beats", []))
    payload = {
        "source": "llm_chunked",
        "complete": complete,
        "completed_chunks": len(chunks),
        "chunks": chunks,
        "beats": _dedupe_beats(all_beats),
    }
    write_json(cache_path, payload)


def _write_partial_highlights(cache_path: Optional[str], batches: List[Dict], complete: bool = False) -> None:
    if not cache_path:
        return
    highlights: List[Dict] = []
    for batch in sorted(batches, key=lambda item: int(item.get("index", 0) or 0)):
        highlights.extend(batch.get("highlights", []))
    payload = {
        "version": HIGHLIGHT_SELECTION_VERSION,
        "complete": complete,
        "completed_batches": len(batches),
        "batches": batches,
        "highlights": dedupe_highlights(highlights),
    }
    write_json(cache_path, payload)


def build_beat_map(
    transcript: Dict,
    llm_fn: LLMFn,
    analysis_map: Optional[Dict] = None,
    episode_digest: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> Dict:
    duration = float(transcript.get("duration", 0.0) or 0.0)
    chunk_plan = _beat_map_chunks(transcript)
    total_chunks = len(chunk_plan)
    existing = read_json(cache_path) if cache_path else None
    completed_by_index = _partial_chunks(existing)
    completed_chunks = [completed_by_index[i] for i in sorted(completed_by_index)]

    if existing and existing.get("complete") and existing.get("beats"):
        user_log("Story beats ready", f"{len(existing.get('beats', []))} beats loaded from cache")
        return existing

    progress = Progress("LLM story beats", total_chunks)
    for done_index in sorted(completed_by_index):
        progress.update(
            done_index,
            f"chunk {done_index}/{total_chunks}: cached",
            force=True,
        )

    pending_chunks = [chunk for chunk in chunk_plan if int(chunk["index"]) not in completed_by_index]

    def analyze_chunk(chunk: Dict) -> Dict:
        chunk_index = int(chunk["index"])
        start = float(chunk["start"])
        end = float(chunk["end"])
        text = _transcript_text_for_window(transcript, start, end)
        if not text.strip():
            return {
                "index": chunk_index,
                "start": start,
                "end": end,
                "beats": [],
                "skipped": "no speech",
            }
        analysis_json = json.dumps(
            _analysis_for_window(analysis_map, start, end),
            ensure_ascii=False,
            separators=(",", ":"),
        )[:9000]
        prompt = (
            f"{BEAT_MAP_PROMPT}\n\n"
            f"Episode editorial brief:\n{_episode_context_text(episode_digest)}\n\n"
            f"Video window: {start:.0f}s-{end:.0f}s of {duration:.0f}s\n\n"
            f"Local media analysis for this window:\n{analysis_json}\n\n"
            f"Timestamped transcript window:\n{text[:18000]}"
        )
        user_log(
            "GPT story map",
            f"reading transcript chunk {chunk_index}/{total_chunks} ({start:.0f}s-{end:.0f}s)",
        )
        try:
            raw = llm_fn(prompt)
            beats = _parse_beats(_parse_json_loose(raw))
        except Exception as e:
            raise RuntimeError(
                f"Story beat analysis failed on chunk {chunk_index} ({start:.0f}s-{end:.0f}s): {e}"
            ) from e
        return {
            "index": chunk_index,
            "start": start,
            "end": end,
            "beats": beats,
        }

    workers = _llm_parallelism(len(pending_chunks))
    if pending_chunks:
        user_log(
            "GPT story map",
            f"{len(pending_chunks)} chunks left, {workers} request{'s' if workers != 1 else ''} in parallel",
        )
    if workers <= 1:
        for chunk in pending_chunks:
            result = analyze_chunk(chunk)
            completed_chunks.append(result)
            detail = "no speech" if result.get("skipped") else f"{len(result.get('beats', []))} beats"
            progress.update(int(result["index"]), f"chunk {int(result['index'])}/{total_chunks}: {detail}", force=True)
            _write_partial_beat_map(cache_path, completed_chunks, complete=False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_chunk, chunk): chunk for chunk in pending_chunks}
            for future in as_completed(futures):
                result = future.result()
                completed_chunks.append(result)
                detail = "no speech" if result.get("skipped") else f"{len(result.get('beats', []))} beats"
                progress.update(int(result["index"]), f"chunk {int(result['index'])}/{total_chunks}: {detail}", force=True)
                _write_partial_beat_map(cache_path, completed_chunks, complete=False)

    all_beats: List[Dict] = []
    for chunk in completed_chunks:
        all_beats.extend(chunk.get("beats", []))
    all_beats = _dedupe_beats(all_beats)
    if not all_beats:
        raise RuntimeError("Story beat analysis returned zero beats.")
    complete_payload = {
        "beats": all_beats,
        "source": "llm_chunked",
        "complete": True,
        "completed_chunks": len(completed_chunks),
        "chunks": sorted(completed_chunks, key=lambda item: int(item.get("index", 0) or 0)),
    }
    if cache_path:
        write_json(cache_path, complete_payload)
    return complete_payload


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = None,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate 0-{min_clips} highlights. Return fewer, or zero, if the moments are not truly strong.",
    )
    full_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    raw = llm_fn(full_prompt)
    return _parse_json_loose(raw)


def call_highlight_api_from_beats(
    beat_map: Dict,
    content_info: Dict,
    num_clips: int,
    llm_fn: LLMFn = None,
    review_llm_fn: Optional[LLMFn] = None,
    analysis_map: Optional[Dict] = None,
    episode_digest: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> Dict:
    beats = beat_map.get("beats", [])
    if not beats:
        return {"highlights": []}

    candidates: List[Dict] = []
    sorted_beats = sorted(beats, key=lambda b: int(b.get("clip_potential", 0) or 0), reverse=True)
    total_batches = max(1, int(math.ceil(len(sorted_beats) / BEAT_SELECT_BATCH_SIZE)))
    existing = read_json(cache_path) if cache_path else None
    completed_by_index = _partial_batches(existing)
    completed_batches = [completed_by_index[i] for i in sorted(completed_by_index)]
    if (
        existing
        and existing.get("complete")
        and existing.get("highlights") is not None
        and existing.get("version") == HIGHLIGHT_SELECTION_VERSION
    ):
        return existing
    for batch in completed_batches:
        candidates.extend(batch.get("highlights", []))
    progress = Progress("LLM candidates", total_batches)
    for done_index in sorted(completed_by_index):
        progress.update(done_index, f"batch {done_index}/{total_batches}: cached", force=True)

    batch_plan = []
    for batch_index, offset in enumerate(range(0, len(sorted_beats), BEAT_SELECT_BATCH_SIZE), 1):
        batch = sorted_beats[offset: offset + BEAT_SELECT_BATCH_SIZE]
        if not batch:
            continue
        if batch_index in completed_by_index:
            continue
        batch_plan.append({"index": batch_index, "beats": batch})

    def analyze_batch(item: Dict) -> Dict:
        batch_index = int(item["index"])
        batch = item["beats"]
        batch_json = json.dumps({"beats": batch}, ensure_ascii=False, separators=(",", ":"))[:14000]
        prompt = BEAT_HIGHLIGHT_PROMPT.format(
            virality_criteria=VIRALITY_CRITERIA,
            content_type=content_info.get("content_type", "other"),
            density=content_info.get("density", "medium"),
            episode_context=_episode_context_text(episode_digest),
            num_clips_instruction="Generate 0-5 highlights from this batch. Return zero only if the batch is genuinely unusable.",
            beat_map_json=batch_json,
        )
        try:
            user_log("GPT clip candidates", f"batch {batch_index}/{total_batches}, reviewing {len(batch)} story beats")
            result = _parse_json_loose(llm_fn(prompt))
            batch_candidates = result.get("highlights", [])
            return {
                "index": batch_index,
                "highlights": batch_candidates,
            }
        except Exception as e:
            raise RuntimeError(f"Candidate selection failed on beat batch {batch_index}: {e}") from e

    workers = _llm_parallelism(len(batch_plan))
    if batch_plan:
        user_log(
            "GPT clip candidates",
            f"{len(batch_plan)} batches left, {workers} request{'s' if workers != 1 else ''} in parallel",
        )
    if workers <= 1:
        for item in batch_plan:
            result = analyze_batch(item)
            batch_candidates = result.get("highlights", [])
            candidates.extend(batch_candidates)
            completed_batches.append(result)
            progress.update(int(result["index"]), f"batch {int(result['index'])}/{total_batches}: {len(batch_candidates)} candidates", force=True)
            _write_partial_highlights(cache_path, completed_batches, complete=False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_batch, item): item for item in batch_plan}
            for future in as_completed(futures):
                result = future.result()
                batch_candidates = result.get("highlights", [])
                candidates.extend(batch_candidates)
                completed_batches.append(result)
                progress.update(int(result["index"]), f"batch {int(result['index'])}/{total_batches}: {len(batch_candidates)} candidates", force=True)
                _write_partial_highlights(cache_path, completed_batches, complete=False)

    candidates = keep_postable_highlights(
        dedupe_highlights(candidates),
        min_score=60,
        min_ending=45,
        max_context_risk=95,
        min_first_3s=40,
        min_payoff=40,
        min_shareability=35,
    )
    if not candidates:
        candidates = _fallback_highlights_from_beats(
            sorted_beats,
            num_clips,
            reason="LLM candidate batches returned zero usable clips",
        )
        user_log("GPT clip candidates", f"LLM returned 0 usable candidates; using {len(candidates)} best story beats")
    if not candidates:
        result = {"version": HIGHLIGHT_SELECTION_VERSION, "highlights": [], "complete": True, "batches": completed_batches}
        if cache_path:
            write_json(cache_path, result)
        return result

    top_candidates = sorted(candidates, key=lambda h: int(h.get("score", 0) or 0), reverse=True)[: max(num_clips * 5, 20)]
    review_items = []
    for index, item in enumerate(top_candidates, 1):
        review_items.append({
            "id": index,
            "title": item.get("title", ""),
            "start_time": float(item.get("start_time", 0.0)),
            "end_time": float(item.get("end_time", 0.0)),
            "score": int(item.get("score", 0) or 0),
            "hook_sentence": item.get("hook_sentence", ""),
            "virality_reason": item.get("virality_reason", ""),
        })
    prompt = FINAL_RERANK_PROMPT.format(
        num_clips=num_clips,
        candidates_json=json.dumps(review_items, ensure_ascii=False, separators=(",", ":"))[:18000],
    )
    user_log("LLM final review", f"choosing best clips from {len(top_candidates)} candidates (AUTO max {num_clips})")
    data = _parse_json_loose((review_llm_fn or llm_fn)(prompt))
    selected = []
    by_id = {i + 1: h for i, h in enumerate(top_candidates)}
    for item in data.get("selected", []):
        cid = int(item.get("id", 0) or 0)
        if cid not in by_id:
            continue
        h = dict(by_id[cid])
        if "start_time" in item:
            h["start_time"] = float(item["start_time"])
        if "end_time" in item:
            h["end_time"] = float(item["end_time"])
        if "score" in item:
            h["score"] = int(item["score"])
        if "reason" in item:
            h["virality_reason"] = str(item["reason"])
        if int(h.get("score", 0) or 0) >= 65:
            selected.append(h)
    if not selected:
        selected = top_candidates[: max(1, min(num_clips, 4))]
        user_log("LLM final review", f"returned 0 clips; using {len(selected)} best candidates instead")
    result = {
        "version": HIGHLIGHT_SELECTION_VERSION,
        "highlights": keep_postable_highlights(
            selected,
            min_score=62,
            min_ending=45,
            max_context_risk=94,
            min_first_3s=40,
            min_payoff=40,
            min_shareability=35,
        ),
        "complete": True,
        "batches": sorted(completed_batches, key=lambda item: int(item.get("index", 0) or 0)),
    }
    if cache_path:
        write_json(cache_path, result)
    return result


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def keep_postable_highlights(
    highlights: List[Dict],
    min_score: int = MIN_POSTABLE_SCORE,
    max_duration: float = 75.5,
    min_ending: int = 72,
    max_context_risk: int = 82,
    min_first_3s: int = 60,
    min_payoff: int = 60,
    min_shareability: int = 55,
) -> List[Dict]:
    kept = []
    for h in highlights:
        try:
            start = float(h.get("start_time", 0.0))
            end = float(h.get("end_time", 0.0))
            score = int(h.get("score", 0) or 0)
        except (TypeError, ValueError):
            continue
        duration = end - start
        viral_score = int(h.get("viral_score", score) or score)
        effective_score = max(score, viral_score)
        if effective_score < min_score:
            continue
        matrix = h.get("score_matrix") if isinstance(h.get("score_matrix"), dict) else {}
        try:
            ending = int(matrix.get("ending_completeness", 100))
            context_risk = int(matrix.get("context_loss_risk", 0))
            first_3s = int(matrix.get("first_3s", matrix.get("hook", 100)))
            payoff = int(matrix.get("payoff_strength", 100))
            shareability = int(matrix.get("shareability", 100))
        except (TypeError, ValueError):
            ending = 100
            context_risk = 0
            first_3s = 100
            payoff = 100
            shareability = 100
        if (
            ending < min_ending
            or context_risk > max_context_risk
            or first_3s < min_first_3s
            or payoff < min_payoff
            or shareability < min_shareability
        ):
            continue
        if duration < 10.0 or duration > max_duration:
            continue
        kept.append(h)
    return kept


def _fallback_quality_clip(h: Dict, index: int) -> Dict:
    clip = dict(h)
    base_title = str(clip.get("title") or f"Clip {index}")
    score = int(clip.get("viral_score", clip.get("score", 70)) or 70)
    clip["viral_score"] = max(score, 70)
    clip["score"] = max(int(clip.get("score", 0) or 0), clip["viral_score"])
    clip.setdefault("semantic_key", _compact_text(base_title.casefold(), 90))
    clip.setdefault("intro_overlay", "")
    clip.setdefault("hook_sentence", str(clip.get("hook_sentence") or base_title))
    clip.setdefault("pause_policy", "balanced")
    clip.setdefault("highlight_keywords", [])
    clip.setdefault("titles", [base_title, base_title[:70], base_title[:55]])
    clip.setdefault("description", str(clip.get("virality_reason") or "A complete short-worthy moment from the episode."))
    clip.setdefault("hashtags", ["#shorts"])
    clip.setdefault("pinned_comment", "Which moment was better?")
    clip.setdefault("score_matrix", {
        "hook": 70,
        "first_3s": 70,
        "standalone_clarity": 70,
        "setup_completeness": 70,
        "payoff_strength": 70,
        "ending_completeness": 70,
        "shareability": 70,
        "rewatch_potential": 65,
        "context_loss_risk": 25,
    })
    if not clip.get("virality_reason"):
        clip["virality_reason"] = "Kept by fallback because the episode has usable candidates and quality gate should not zero the batch."
    return clip


def final_quality_review_with_llm(
    transcript: Dict,
    candidates: List[Dict],
    num_clips: int,
    llm_fn: LLMFn,
    episode_digest: Optional[Dict] = None,
    analysis_map: Optional[Dict] = None,
    content_info: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> List[Dict]:
    if not candidates or num_clips <= 0:
        return []
    cached = read_json(cache_path) if cache_path else None
    if cached and cached.get("complete") and cached.get("clips") is not None and cached.get("version") == FINAL_QUALITY_REVIEW_VERSION:
        return cached.get("clips", [])[:num_clips]

    review_items = []
    source_candidates = sorted(
        candidates,
        key=lambda h: int(h.get("viral_score", h.get("score", 0)) or 0),
        reverse=True,
    )[: max(num_clips * 4, 16)]
    for index, h in enumerate(source_candidates, 1):
        start = float(h.get("start_time", 0.0))
        end = float(h.get("end_time", 0.0))
        context = _transcript_text_for_window(transcript, max(0.0, start - 8.0), end + 10.0)
        media = _analysis_for_window(analysis_map, start, end)
        review_items.append({
            "id": index,
            "title": h.get("title", ""),
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "score": int(h.get("score", 0) or 0),
            "viral_score": int(h.get("viral_score", h.get("score", 0)) or 0),
            "hook_sentence": h.get("hook_sentence", ""),
            "virality_reason": h.get("virality_reason", ""),
            "score_matrix": h.get("score_matrix", {}),
            "transcript_context": context[:2400],
            "media": media,
        })

    content_info = content_info or {"content_type": "other", "density": "medium"}
    prompt = FINAL_QUALITY_PROMPT.format(
        episode_context=_episode_context_text(episode_digest),
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        candidates_json=json.dumps(review_items, ensure_ascii=False, separators=(",", ":"))[:26000],
    )
    user_log("Final quality review", f"scoring {len(review_items)} clips for hook, payoff, duplicates, and upload metadata")
    data = _parse_json_loose(llm_fn(prompt))
    by_id = {i + 1: h for i, h in enumerate(source_candidates)}
    selected: List[Dict] = []
    seen_keys = set()
    for item in data.get("clips", []):
        try:
            cid = int(item.get("id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cid not in by_id or item.get("keep") is False:
            continue
        h = dict(by_id[cid])
        try:
            h["start_time"] = float(item.get("start_time", h.get("start_time", 0.0)))
            h["end_time"] = float(item.get("end_time", h.get("end_time", 0.0)))
        except (TypeError, ValueError):
            continue
        if h["end_time"] <= h["start_time"]:
            continue
        semantic_key = _compact_text(str(item.get("semantic_key", h.get("title", ""))).casefold(), 90)
        if semantic_key and semantic_key in seen_keys:
            continue
        if semantic_key:
            seen_keys.add(semantic_key)
            h["semantic_key"] = semantic_key
        try:
            h["viral_score"] = int(item.get("viral_score", h.get("score", 0)) or 0)
        except (TypeError, ValueError):
            h["viral_score"] = int(h.get("score", 0) or 0)
        h["score"] = max(int(h.get("score", 0) or 0), int(h.get("viral_score", 0) or 0))
        if isinstance(item.get("score_matrix"), dict):
            h["score_matrix"] = item["score_matrix"]
        for key in ("intro_overlay", "hook_sentence", "pause_policy", "description", "pinned_comment", "reason"):
            if item.get(key) is not None:
                h[key] = str(item.get(key) or "")
        for key in ("highlight_keywords", "titles", "hashtags"):
            value = item.get(key)
            if isinstance(value, list):
                h[key] = [str(v) for v in value if str(v).strip()]
        if h.get("titles"):
            h["title"] = h["titles"][0]
        if h.get("reason"):
            h["virality_reason"] = h["reason"]
        selected.append(h)

    selected = keep_postable_highlights(
        dedupe_highlights(selected),
        min_score=68,
        min_ending=50,
        max_context_risk=92,
        min_first_3s=45,
        min_payoff=45,
        min_shareability=40,
    )
    if not selected and source_candidates:
        fallback_count = min(num_clips, max(1, min(3, len(source_candidates))))
        selected = [_fallback_quality_clip(h, i + 1) for i, h in enumerate(source_candidates[:fallback_count])]
        user_log("Final quality review", f"quality gate kept 0, using {len(selected)} best candidates instead of skipping the episode")
    selected = sorted(selected, key=lambda h: int(h.get("viral_score", h.get("score", 0)) or 0), reverse=True)[:num_clips]
    user_log("Final quality review", f"{len(selected)} clips passed the upload-quality gate")
    if cache_path:
        write_json(cache_path, {"complete": True, "version": FINAL_QUALITY_REVIEW_VERSION, "clips": selected})
    return selected


def refine_highlight_boundaries_with_llm(
    transcript: Dict,
    highlights: List[Dict],
    llm_fn: LLMFn,
    episode_digest: Optional[Dict] = None,
) -> List[Dict]:
    if not highlights:
        return []

    segments = transcript.get("segments", [])
    review_items = []
    for index, h in enumerate(highlights, 1):
        start = float(h.get("start_time", 0.0))
        end = float(h.get("end_time", 0.0))
        context_segments = []
        for segment in segments:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", 0.0))
            if seg_end < start - 14.0 or seg_start > end + 22.0:
                continue
            text = str(segment.get("text", "")).strip()
            if text:
                context_segments.append({
                    "start": round(seg_start, 2),
                    "end": round(seg_end, 2),
                    "text": text,
                })
        review_items.append({
            "id": index,
            "title": h.get("title", ""),
            "current_start": round(start, 2),
            "current_end": round(end, 2),
            "score": int(h.get("score", 0) or 0),
            "reason": h.get("virality_reason", ""),
            "context_segments": context_segments[:42],
        })

    prompt = f"""You are a semantic boundary editor for short-form clips from a scripted episode.

The clips are already selected. Do NOT replace them with different scenes.
Your only job is to adjust start_time/end_time so each clip is a complete mini-scene.

Episode editorial brief:
{_episode_context_text(episode_digest)}

Rules:
- Keep the selected idea, but include the necessary setup, turn, payoff, final answer, reaction, or visual/silent landing.
- Do not end just because speech pauses; end when the meaning of the scene beat has landed.
- Do not cut mid-thought, before the punchline/rebuttal/answer, or before the reaction that makes the joke work.
- Remove unrelated lead-in/outro only when it is clearly outside the selected beat.
- Aim for 18-60 seconds. Use up to 75 seconds when needed for a complete ending.
- If a candidate cannot be made complete from the given context, set keep=false.
- Put boundaries on transcript segment boundaries when possible.
- Prefer text anchors over raw timestamps: choose exact transcript words where the clip begins and ends.

Respond JSON only:
{{"clips":[{{"id":int,"keep":true,"start_time":float,"end_time":float,"start_anchor_text":"exact transcript words","end_anchor_text":"exact transcript words","include_after_end_seconds":0.0,"score_matrix":{{"hook":0,"standalone_clarity":0,"setup_completeness":0,"payoff_strength":0,"ending_completeness":0,"context_loss_risk":0}},"reason":"short boundary reason"}}]}}

Candidates:
{json.dumps(review_items, ensure_ascii=False, separators=(",", ":"))[:24000]}
"""
    user_log("Semantic boundary review", f"adjusting {len(review_items)} selected clips")
    data = _parse_json_loose(llm_fn(prompt))
    by_id = {i + 1: h for i, h in enumerate(highlights)}
    refined = []
    for item in data.get("clips", []):
        try:
            cid = int(item.get("id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cid not in by_id or item.get("keep") is False:
            continue
        h = dict(by_id[cid])
        try:
            start = float(item.get("start_time", h.get("start_time", 0.0)))
            end = float(item.get("end_time", h.get("end_time", 0.0)))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        h["start_time"] = start
        h["end_time"] = end
        if item.get("start_anchor_text"):
            h["start_anchor_text"] = str(item.get("start_anchor_text"))
        if item.get("end_anchor_text"):
            h["end_anchor_text"] = str(item.get("end_anchor_text"))
        try:
            h["include_after_end_seconds"] = max(0.0, min(float(item.get("include_after_end_seconds", 0.0) or 0.0), 3.0))
        except (TypeError, ValueError):
            pass
        if isinstance(item.get("score_matrix"), dict):
            h["score_matrix"] = item["score_matrix"]
        if item.get("reason"):
            h["boundary_reason"] = str(item.get("reason"))
        refined.append(h)
    if not refined:
        user_log("Semantic boundary review", "kept original boundaries")
        return highlights
    user_log("Semantic boundary review", f"{len(refined)} clips adjusted/kept")
    return refined


def verify_highlights_with_llm(
    transcript: Dict,
    candidates: List[Dict],
    num_clips: int,
    llm_fn: LLMFn,
) -> List[Dict]:
    if not candidates or num_clips <= 0:
        return []

    segments = transcript.get("segments", [])
    review_items = []
    for index, h in enumerate(candidates[: max(num_clips * 3, num_clips)], 1):
        start = float(h.get("start_time", 0.0))
        end = float(h.get("end_time", 0.0))
        context = " ".join(
            s.get("text", "").strip()
            for s in segments
            if float(s.get("end", 0.0)) >= start - 10 and float(s.get("start", 0.0)) <= end + 10
        )[:1800]
        review_items.append({
            "id": index,
            "title": h.get("title", ""),
            "start_time": start,
            "end_time": end,
            "score": int(h.get("score", 0)),
            "hook_sentence": h.get("hook_sentence", ""),
            "virality_reason": h.get("virality_reason", ""),
            "transcript_context": context,
        })

    prompt = f"""You are the final editor choosing complete short clips.
You are given candidate clips with transcript context. Your job:
1. SELECT up to {num_clips} clips
2. ADJUST timestamps only when it improves clarity, completeness, or removes unrelated edges

Rules:
- QUALITY GATE: {num_clips} is the maximum, not a quota. Select fewer clips if only fewer are genuinely strong.
- POSTABLE ONLY: reject clips that are merely okay, mildly funny, unclear, slow, or included only to fill the count.
- COMPLETE ENDING: keep the payoff, answer, reaction, or final sentence. Do not cut mid-thought.
- VISUAL LANDING: for scripted scenes/cartoons, keep the silent reaction or visual gag landing after the last spoken line when it completes the moment.
- CLEAN START: start where the viewer has enough context and the hook begins.
- DURATION: usually 10-60s, preferably 18-45s if that keeps the full idea intact. 61-75s is acceptable only to preserve the final sentence/reaction.
- If a candidate cannot fit under 75 seconds without cutting the ending, move start_time later to a smaller complete sub-beat or reject it.
- Put end_time on a phrase/segment boundary, not in the middle of spoken text.
- ONE IDEA PER CLIP: each selected clip should cover one coherent beat.
- REJECT: boring filler, repeated ideas, unclear context, clips that need missing setup, clips that end before the idea lands, or clips without a strong turn/payoff/reaction.

For each selected clip, you can adjust start_time and end_time to make it more complete and watchable.
Respond JSON only: {{"selected":[{{"id":int,"start_time":float,"end_time":float}}], "notes":"short reason"}}

Candidates:
{json.dumps(review_items, ensure_ascii=False, indent=2)}
"""
    raw = llm_fn(prompt)
    data = _parse_json_loose(raw)
    selected_items = data.get("selected", [])
    by_id = {i + 1: h for i, h in enumerate(candidates)}
    selected = []
    for item in selected_items:
        cid = int(item.get("id", 0))
        if cid in by_id:
            h = dict(by_id[cid])
            if "start_time" in item:
                h["start_time"] = float(item["start_time"])
            if "end_time" in item:
                h["end_time"] = float(item["end_time"])
        if int(h.get("score", 0) or 0) >= MIN_POSTABLE_SCORE:
            selected.append(h)
    user_log("Final boundary check", f"{len(selected)} clips selected")
    return selected[:num_clips]


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
    review_llm_fn: Optional[LLMFn] = None,
    beat_map: Optional[Dict] = None,
    analysis_map: Optional[Dict] = None,
    episode_digest: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` is required — pass in an OpenAI-backed callable.
    """
    if llm_fn is None:
        raise ValueError("llm_fn is required (pass call_openai_llm)")
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    user_log(
        "Content profile",
        f"{content_info.get('content_type')} / {content_info.get('density')} / {duration:.0f}s",
    )

    if beat_map and beat_map.get("beats"):
        user_log("Beat map selection", f"{len(beat_map.get('beats', []))} beats to review")
        result = call_highlight_api_from_beats(
            beat_map,
            content_info,
            num_clips=num_clips,
            llm_fn=llm_fn,
            review_llm_fn=review_llm_fn,
            analysis_map=analysis_map,
            episode_digest=episode_digest,
            cache_path=cache_path,
        )
        highlights = dedupe_highlights(result.get("highlights", []))
        if not highlights:
            highlights = _fallback_highlights_from_beats(
                beat_map.get("beats", []),
                num_clips,
                reason="Beat-map selection returned zero clips",
            )
            user_log("Beat map selection", f"using {len(highlights)} fallback clips from best story beats")
        return {"highlights": highlights, "beat_map_source": beat_map.get("source")}

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(text, content_info, chunk["duration"], num_clips=num_clips, is_chunk=True, llm_fn=llm_fn)
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn)
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}
