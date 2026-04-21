"""Orchestrates the full wayfinding pipeline: transcribe -> parse intent -> retrieve -> speak."""

from dataclasses import dataclass

from core.config import get_logger
from core.db import query_by_pois, query_by_text
from core.graph import AirportGraph
from core.intent import parse_intent
from core.transcribe import transcribe_audio
from core.tts import text_to_speech

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    transcript: str
    start_poi: str
    end_poi: str
    route_text: str
    audio_path: str | None
    floor_info: str = ""
    error: str = ""


NO_ROUTE_MSG = (
    "Sorry, I don't have directions for that route yet. "
    "Try asking about a different pair of locations."
)

MISSING_INTENT_MSG = (
    "I couldn't understand your starting point or destination. "
    "Please say something like: How do I get from H&M to Check-in 2?"
)

_graph: AirportGraph | None = None


def set_graph(graph: AirportGraph) -> None:
    global _graph
    _graph = graph


def _resolve_poi_name(name: str) -> str:
    """Match a spoken name to a graph POI. Returns canonical name or original input."""
    if _graph is None:
        return name
    poi = _graph.find_poi_by_name(name)
    if poi:
        return poi.name
    log.warning("Could not resolve POI name: '%s'", name)
    return name


def _get_floor_info(start_name: str, end_name: str) -> str:
    if _graph is None:
        return ""
    start_poi = _graph.find_poi_by_name(start_name)
    end_poi = _graph.find_poi_by_name(end_name)
    if not start_poi or not end_poi:
        return ""
    if start_poi.floor == end_poi.floor:
        return f"Same floor (Floor {start_poi.floor})"
    return f"Floor {start_poi.floor} to Floor {end_poi.floor}"


def _retrieve_route(start: str, end: str, raw_query: str) -> str:
    route = query_by_pois(start, end)
    if route:
        log.info("Exact metadata match: %s -> %s", start, end)
        return route

    route = query_by_text(raw_query)
    if route:
        log.info("Semantic fallback match for: %s", raw_query[:80])
        return route

    log.warning("No route found: %s -> %s", start, end)
    return NO_ROUTE_MSG


def run_voice(audio_path: str) -> PipelineResult:
    """Full pipeline from audio input."""
    log.info("Voice pipeline started")
    transcript = transcribe_audio(audio_path)
    log.info("Transcription: %s", transcript)
    return _run_from_text(transcript)


def run_text(text: str) -> PipelineResult:
    """Full pipeline from text input (skips Whisper)."""
    log.info("Text pipeline: %s", text[:80])
    return _run_from_text(text)


def _run_from_text(text: str) -> PipelineResult:
    intent = parse_intent(text)
    start = intent["start"]
    end = intent["end"]
    log.info("Parsed intent: %s -> %s", start, end)

    if start == "unknown" or end == "unknown":
        return PipelineResult(
            transcript=text,
            start_poi=start,
            end_poi=end,
            route_text=MISSING_INTENT_MSG,
            audio_path=None,
        )

    display_start = _resolve_poi_name(start)
    display_end = _resolve_poi_name(end)
    floor_info = _get_floor_info(display_start, display_end)
    route_text = _retrieve_route(display_start, display_end, text)

    audio_path = None
    error = ""
    try:
        audio_path = text_to_speech(route_text)
    except Exception as exc:
        error = f"TTS failed: {exc}"
        log.error("TTS generation failed: %s", exc)

    return PipelineResult(
        transcript=text,
        start_poi=display_start,
        end_poi=display_end,
        route_text=route_text,
        audio_path=audio_path,
        floor_info=floor_info,
        error=error,
    )
