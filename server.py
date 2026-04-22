"""FastAPI server for Airport Wayfinding - clean API replacing Gradio."""

import json
import shutil
import tempfile
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import get_logger
from core.db import seed_from_json, get_collection, get_route_count, reset_collection
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


# --- Admin endpoints ---

MAPS_DIR = Path("data/maps")


@app.get("/api/admin/maps")
def list_maps():
    """List all uploaded floor plan images."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    maps = []
    for f in sorted(MAPS_DIR.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            maps.append({
                "filename": f.name,
                "url": f"/api/admin/maps/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return maps


@app.get("/api/admin/maps/{filename}")
def serve_map(filename: str):
    """Serve a floor plan image."""
    path = MAPS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Map not found")
    media = "image/png" if path.suffix == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


@app.post("/api/admin/maps/upload")
async def upload_map(
    file: UploadFile = File(...),
    floor_number: int = Form(0),
    floor_name: str = Form(""),
):
    """Upload a floor plan image and save it."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "map.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Only PNG, JPG, and WebP images are supported")

    safe_name = f"floor_{floor_number}{suffix}"
    dest = MAPS_DIR / safe_name

    content = await file.read()
    dest.write_bytes(content)
    log.info("Saved map: %s (%d KB)", safe_name, len(content) // 1024)

    return {
        "filename": safe_name,
        "url": f"/api/admin/maps/{safe_name}",
        "floor_number": floor_number,
        "size_kb": round(len(content) / 1024, 1),
    }


@app.post("/api/admin/extract")
async def extract_pois(
    file: UploadFile = File(...),
    floor_number: int = Form(0),
):
    """Run VLM extraction on a floor plan image."""
    from core.vlm import extract_and_compare

    suffix = Path(file.filename or "map.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        vlm_data, comparison = extract_and_compare(
            image_path=tmp.name,
            floor_number=floor_number,
        )

        return {
            "vlm_data": vlm_data,
            "comparison": comparison,
        }
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.post("/api/admin/compile")
def compile_routes():
    """Run the full compiler: generate segments via LLM, chain routes, push to ChromaDB."""
    from core.compiler import run_compiler

    try:
        reset_collection()
        stats = run_compiler()
        log.info("Compilation complete: %s", stats)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        log.error("Compilation failed: %s", e)
        raise HTTPException(500, f"Compilation failed: {e}")


@app.get("/api/admin/config")
def get_config():
    """Return the current airport config for display."""
    config_path = Path("data/airport_config.json")
    if not config_path.exists():
        return {"floors": [], "pois": [], "adjacencies": []}
    return json.loads(config_path.read_text())


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    _init()
    log.info("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
