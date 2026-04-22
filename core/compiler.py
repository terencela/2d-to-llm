"""Stage 1 Compiler: generate route texts for all reachable POI pairs and push to ChromaDB."""

import json
from pathlib import Path

from openai import OpenAI

from core.config import get_logger, get_openai_client
from core.db import get_collection
from core.graph import AirportGraph, load_graph

log = get_logger("compiler")


SEGMENT_PROMPT = """You write short walking directions inside an airport.
Given two adjacent points of interest, write exactly 1-2 sentences describing
how to walk from the start to the end.

Be specific: mention landmarks, turns, distances, and floor changes.
Use natural language a traveler would understand.
Do NOT mention the start location name at the beginning (the caller chains segments).
Start with an action verb (walk, turn, take, continue, follow, head, proceed).

Context:
- Connection type: {edge_type}
- Approximate distance: {distance_m} meters
{notes_line}"""


def generate_segment_text(
    client: OpenAI,
    start_name: str,
    end_name: str,
    edge_type: str,
    distance_m: int,
    notes: str = "",
) -> str:
    """Generate a 1-2 sentence walking direction for one adjacent pair."""
    notes_line = f"- Note: {notes}" if notes else ""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=150,
        messages=[
            {"role": "system", "content": SEGMENT_PROMPT.format(
                edge_type=edge_type,
                distance_m=distance_m,
                notes_line=notes_line,
            )},
            {"role": "user", "content": f"From: {start_name}\nTo: {end_name}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def compile_segments(graph: AirportGraph) -> dict[str, str]:
    """Generate route text for every adjacent pair. Returns {edge_key: text}."""
    client = get_openai_client()
    segments: dict[str, str] = {}
    pairs = graph.get_adjacent_pairs()

    for i, (from_id, to_id) in enumerate(pairs):
        edge = graph.get_edge(from_id, to_id)
        if not edge:
            continue

        from_poi = graph.pois[from_id]
        to_poi = graph.pois[to_id]
        key = f"{from_id}|{to_id}"

        text = generate_segment_text(
            client=client,
            start_name=from_poi.name,
            end_name=to_poi.name,
            edge_type=edge.edge_type,
            distance_m=edge.distance_m,
            notes=edge.notes,
        )
        segments[key] = text
        log.info("[%d/%d] %s -> %s", i + 1, len(pairs), from_poi.name, to_poi.name)

    return segments


def chain_segments(
    graph: AirportGraph,
    segments: dict[str, str],
    path: list[str],
) -> str:
    """Chain adjacent segment texts along a path into a full route."""
    if len(path) < 2:
        return ""

    start_poi = graph.pois[path[0]]
    parts = [f"From {start_poi.name}:"]

    for i in range(len(path) - 1):
        key = f"{path[i]}|{path[i+1]}"
        segment = segments.get(key)
        if segment:
            parts.append(segment)
        else:
            to_poi = graph.pois[path[i+1]]
            parts.append(f"Continue to {to_poi.name}.")

    return " ".join(parts)


def compile_all_routes(
    graph: AirportGraph,
    segments: dict[str, str],
) -> list[dict]:
    """Build full route texts for every reachable A-to-B pair."""
    routes = []
    reachable = graph.get_all_reachable_pairs()

    for start_id, end_id, path in reachable:
        start_poi = graph.pois[start_id]
        end_poi = graph.pois[end_id]

        if start_poi.poi_type == "vertical" or end_poi.poi_type == "vertical":
            continue

        route_text = chain_segments(graph, segments, path)
        routes.append({
            "start": start_poi.name.lower(),
            "end": end_poi.name.lower(),
            "start_id": start_id,
            "end_id": end_id,
            "route_text": route_text,
            "path": path,
            "floors_crossed": len({graph.pois[p].floor for p in path}),
        })

    return routes


def push_to_chromadb(routes: list[dict]) -> int:
    """Upsert compiled routes into ChromaDB. Returns count."""
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for route in routes:
        route_id = f"{route['start']}|{route['end']}"
        ids.append(route_id)
        documents.append(route["route_text"])
        metadatas.append({
            "start_poi": route["start"],
            "end_poi": route["end"],
            "start_id": route["start_id"],
            "end_id": route["end_id"],
            "floors_crossed": route["floors_crossed"],
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def save_compiled_data(
    segments: dict[str, str],
    routes: list[dict],
    output_dir: str = "data",
) -> None:
    """Save compiled segments and routes to JSON for inspection."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    Path(f"{output_dir}/compiled_segments.json").write_text(
        json.dumps(segments, indent=2)
    )
    Path(f"{output_dir}/compiled_routes.json").write_text(
        json.dumps(routes, indent=2)
    )


def run_compiler(config_path: str = "data/airport_config.json") -> dict:
    """Full compilation pipeline. Returns summary stats with timing."""
    import time

    t_start = time.monotonic()

    log.info("Loading airport graph...")
    graph = load_graph(config_path)
    log.info("%d POIs, %d edges", len(graph.pois), len(graph.edges))

    t_segments = time.monotonic()
    log.info("Generating segment texts (LLM calls)...")
    segments = compile_segments(graph)
    t_segments_done = time.monotonic()
    segment_sec = round(t_segments_done - t_segments, 1)
    log.info("Generated %d segments in %.1fs (%.1fs/segment)",
             len(segments), segment_sec,
             segment_sec / max(len(segments), 1))

    t_chain = time.monotonic()
    log.info("Chaining routes for all reachable pairs...")
    routes = compile_all_routes(graph, segments)
    t_chain_done = time.monotonic()
    log.info("Built %d full routes in %.1fs", len(routes), t_chain_done - t_chain)

    log.info("Saving compiled data...")
    save_compiled_data(segments, routes)

    t_db = time.monotonic()
    log.info("Pushing to ChromaDB...")
    count = push_to_chromadb(routes)
    t_db_done = time.monotonic()
    log.info("Upserted %d routes in %.1fs", count, t_db_done - t_db)

    total_sec = round(time.monotonic() - t_start, 1)
    log.info("Total compilation: %.1fs", total_sec)

    return {
        "pois": len(graph.pois),
        "edges": len(graph.edges),
        "segments": len(segments),
        "routes": len(routes),
        "stored": count,
        "timing": {
            "segments_sec": segment_sec,
            "avg_segment_sec": round(segment_sec / max(len(segments), 1), 2),
            "chaining_sec": round(t_chain_done - t_chain, 1),
            "db_sec": round(t_db_done - t_db, 1),
            "total_sec": total_sec,
        },
    }
