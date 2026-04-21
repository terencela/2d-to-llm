"""Orchestrates the full wayfinding pipeline: transcribe -> parse intent -> retrieve -> speak."""

from dataclasses import dataclass

from core.db import query_by_pois, query_by_text
from core.intent import parse_intent
from core.transcribe import transcribe_audio
from core.tts import text_to_speech


@dataclass
class PipelineResult:
    transcript: str
    start_poi: str
    end_poi: str
    route_text: str
    audio_path: str | None


NO_ROUTE_MSG = (
    "Sorry, I don't have directions for that route yet. "
    "Try asking about a different pair of locations."
)

MISSING_INTENT_MSG = (
    "I couldn't understand your starting point or destination. "
    "Please say something like: How do I get from H&M to Check-in 2?"
)


def _retrieve_route(start: str, end: str, raw_query: str) -> str:
    """Try metadata match first, fall back to semantic search."""
    route = query_by_pois(start, end)
    if route:
        return route

    route = query_by_text(raw_query)
    if route:
        return route

    return NO_ROUTE_MSG


def run_voice(audio_path: str) -> PipelineResult:
    """Full pipeline from audio input."""
    transcript = transcribe_audio(audio_path)
    return _run_from_text(transcript)


def run_text(text: str) -> PipelineResult:
    """Full pipeline from text input (skips Whisper)."""
    return _run_from_text(text)


def _run_from_text(text: str) -> PipelineResult:
    intent = parse_intent(text)
    start = intent["start"]
    end = intent["end"]

    if start == "unknown" or end == "unknown":
        return PipelineResult(
            transcript=text,
            start_poi=start,
            end_poi=end,
            route_text=MISSING_INTENT_MSG,
            audio_path=None,
        )

    route_text = _retrieve_route(start, end, text)

    audio_path = None
    try:
        audio_path = text_to_speech(route_text)
    except Exception:
        pass

    return PipelineResult(
        transcript=text,
        start_poi=start,
        end_poi=end,
        route_text=route_text,
        audio_path=audio_path,
    )
