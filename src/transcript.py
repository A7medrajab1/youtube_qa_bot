"""
YouTube transcript fetching and preprocessing.
Supports multiple languages with priority order.
"""
import re
from youtube_transcript_api import YouTubeTranscriptApi


YOUTUBE_PATTERNS = [
    r"youtu\.be\/([0-9A-Za-z_-]{11})",
    r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
    r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
    r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
]

# Preferred languages in priority order
PREFERRED_LANGUAGES = ["en", "ar"]


def get_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _pick_best_transcript(transcripts, languages: list[str]):
    """
    Pick the best transcript following this priority:
    1. Manually-created in any preferred language (in order)
    2. Auto-generated in any preferred language (in order)
    3. Any manually-created transcript
    4. Any auto-generated transcript
    """
    manual_by_lang = {}
    generated_by_lang = {}
    any_manual = None
    any_generated = None

    for t in transcripts:
        if t.is_generated:
            generated_by_lang.setdefault(t.language_code, t)
            any_generated = any_generated or t
        else:
            manual_by_lang.setdefault(t.language_code, t)
            any_manual = any_manual or t

    # 1. Manual in preferred language
    for lang in languages:
        if lang in manual_by_lang:
            return manual_by_lang[lang], lang, False

    # 2. Auto-generated in preferred language
    for lang in languages:
        if lang in generated_by_lang:
            return generated_by_lang[lang], lang, True

    # 3. Any manual transcript
    if any_manual:
        return any_manual, any_manual.language_code, False

    # 4. Any auto-generated transcript
    if any_generated:
        return any_generated, any_generated.language_code, True

    return None, None, None


def fetch_transcript(url: str, languages: list[str] = None) -> tuple[list | None, str | None]:
    """
    Fetch the best available transcript for a YouTube video.

    Priority:
    1. Manually-created transcript in preferred languages (en, ar by default)
    2. Auto-generated transcript in preferred languages
    3. Any available transcript

    Args:
        url: YouTube video URL
        languages: List of preferred language codes in priority order
                   (default: ["en", "ar"])

    Returns:
        Tuple of (transcript_data, language_code) or (None, None) if not available.
    """
    if languages is None:
        languages = PREFERRED_LANGUAGES

    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)

    chosen, lang_code, is_generated = _pick_best_transcript(transcripts, languages)
    if not chosen:
        return None, None

    return chosen.fetch(), lang_code


def format_transcript(transcript) -> str:
    """Format transcript snippets into a single string."""
    if not transcript:
        return ""
    lines = []
    for snippet in transcript:
        try:
            lines.append(f"Text: {snippet.text} Start: {snippet.start}")
        except AttributeError:
            continue
    return "\n".join(lines)