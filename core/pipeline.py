"""Orchestrates the full wayfinding pipeline: transcribe -> parse intent -> retrieve -> speak."""

from dataclasses import dataclass

from core.db import query_by_pois, query_by_text
from core.graph import AirportGraph
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
    floor_info: str = ""


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


def _resolve_poi_name(name: str) -> str | None:
    """Try to match a spoken name to a graph POI, return the canonical name."""
    if _graph is None:
        return name
    poi = _graph.find_poi_by_name(name)
    return poi.name if poi else None


def _get_floor_info(start_name: str, end_name: str) -> str:
    """Return floor context string for the route."""
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
    """Try metadata match first, then semantic search, then graph-based chaining."""
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

    resolved_start = _resolve_poi_name(start)
    resolved_end = _resolve_poi_name(end)

    display_start = resolved_start or start
    display_end = resolved_end or end
    floor_info = _get_floor_info(display_start, display_end)

    route_text = _retrieve_route(display_start, display_end, text)

    audio_path = None
    try:
        audio_path = text_to_speech(route_text)
    except Exception:
        pass

    return PipelineResult(
        transcript=text,
        start_poi=display_start,
        end_poi=display_end,
        route_text=route_text,
        audio_path=audio_path,
        floor_info=floor_info,
    )
