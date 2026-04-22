"""FastAPI server for Airport Wayfinding - clean API replacing Gradio."""

import tempfile
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import get_logger
from core.db import seed_from_json, get_collection, get_route_count
from core.graph import load_graph
from core.intent import set_known_pois
from core.pipeline import run_voice, run_text, set_graph

load_dotenv()
log = get_logger("server")

app = FastAPI(title="Airport Wayfinding API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def _init() -> None:
    global _graph
    config_path = "data/airport_config.json"

    if Path(config_path).exists():
        _graph = load_graph(config_path)
        set_graph(_graph)
        set_known_pois(_graph.get_all_poi_names())
        log.info("Graph loaded: %d POIs", len(_graph.pois))

    collection = get_collection()
    if collection.count() == 0 and Path("data/seed_routes.json").exists():
        count = seed_from_json()
        log.info("Seeded %d demo routes", count)


class TextRequest(BaseModel):
    query: str


class DirectionsResponse(BaseModel):
    start_poi: str
    end_poi: str
    floor_info: str
    directions: str
    audio_url: str | None = None
    transcript: str | None = None


class POIInfo(BaseModel):
    name: str
    floor: int
    floor_name: str


@app.get("/api/pois")
def get_pois() -> list[POIInfo]:
    if _graph is None:
        return []

    import json
    config = json.loads(Path("data/airport_config.json").read_text())
    floor_names = {f["id"]: f["name"] for f in config["floors"]}

    result = []
    for poi in _graph.pois.values():
        if poi.poi_type == "vertical":
            continue
        result.append(POIInfo(
            name=poi.name,
            floor=poi.floor,
            floor_name=floor_names.get(poi.floor, f"Floor {poi.floor}"),
        ))

    return sorted(result, key=lambda p: (p.floor, p.name))


@app.post("/api/directions/text")
def directions_text(req: TextRequest) -> DirectionsResponse:
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    result = run_text(req.query)
    audio_url = f"/api/audio/{Path(result.audio_path).name}" if result.audio_path else None

    return DirectionsResponse(
        start_poi=result.start_poi,
        end_poi=result.end_poi,
        floor_info=result.floor_info,
        directions=result.route_text,
        audio_url=audio_url,
    )


@app.post("/api/directions/voice")
async def directions_voice(audio: UploadFile = File(...)) -> DirectionsResponse:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()

        result = run_voice(tmp.name)
        audio_url = f"/api/audio/{Path(result.audio_path).name}" if result.audio_path else None

        return DirectionsResponse(
            transcript=result.transcript,
            start_poi=result.start_poi,
            end_poi=result.end_poi,
            floor_info=result.floor_info,
            directions=result.route_text,
            audio_url=audio_url,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    for d in [tempfile.gettempdir(), "."]:
        path = Path(d) / filename
        if path.exists():
            return FileResponse(path, media_type="audio/mpeg")
    raise HTTPException(404, "Audio file not found")


@app.get("/api/stats")
def get_stats():
    return {
        "routes": get_route_count(),
        "pois": len(_graph.pois) if _graph else 0,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    _init()
    log.info("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
