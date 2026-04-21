# Airport Wayfinding (2D-to-LLM)

Voice and text-based wayfinding for airports. Upload a 2D floor plan, extract POIs via vision LLM, generate route descriptions, and serve them through a voice interface.

## Architecture

```
                    STAGE 1: COMPILER (run once per map)
                    ┌─────────────────────────────────────┐
                    │                                     │
  Floor Plan ──────>│  VLM Extraction ──> POI+Adjacency   │
  (PNG/JPG)         │        │               JSON         │
                    │        v                            │
                    │  LLM Generation Loop                │
                    │  (adjacent pairs only)              │
                    │        │                            │
                    │        v                            │
                    │  BFS Chain ──> ChromaDB             │
                    └─────────────────────────────────────┘

                    STAGE 2: LIVE INTERFACE (per query)
                    ┌─────────────────────────────────────┐
                    │                                     │
  Voice/Text ──────>│  Whisper STT ──> Intent Parser      │
                    │                  (GPT-4o-mini)      │
                    │                      │              │
                    │                      v              │
                    │               ChromaDB Retrieval    │
                    │                      │              │
                    │                      v              │
  Audio <──────────│              OpenAI TTS              │
                    └─────────────────────────────────────┘
```

## Features

- **Multi-floor routing**: 3 floors (Underground, Arrivals, Departures) with elevator/escalator/Skymetro connections
- **25 POIs**: Gates, shops, check-in counters, transport, lounges across Zurich Airport
- **Graph-based pathfinding**: BFS shortest path, adjacent-pair-only LLM generation, programmatic chaining
- **VLM extraction**: Upload a floor plan image, GPT-4o extracts POIs and compares against manual ground truth
- **Voice + text input**: Whisper transcription with text fallback
- **Fuzzy POI matching**: Handles "HM", "checkin 2", "train station" etc.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Run

```bash
python3 app.py
```

Opens a Gradio UI at `http://localhost:7860` with two tabs:

1. **Wayfinding** - ask for directions via voice or text
2. **Map Compiler** - generate routes from airport config, extract POIs from floor plans

## First-time Setup

1. Open the app
2. Go to "Map Compiler" tab
3. Click "Run Compiler" to generate all routes (uses OpenAI API)
4. Switch to "Wayfinding" tab and ask questions

## Project Structure

```
app.py                      # Gradio UI (wayfinding + compiler tabs)
core/
  pipeline.py               # Orchestrates: transcribe -> intent -> retrieve -> speak
  transcribe.py             # Whisper STT
  intent.py                 # GPT-4o-mini intent parser (POI-aware)
  tts.py                    # OpenAI TTS
  db.py                     # ChromaDB init, seed, query
  graph.py                  # POI graph, BFS pathfinding, fuzzy name matching
  compiler.py               # Stage 1: LLM segment generation + route chaining
  vlm.py                    # Vision LLM: floor plan -> POI extraction
data/
  airport_config.json       # Manual POI + adjacency data (25 POIs, 3 floors)
  seed_routes.json          # Hardcoded demo routes (fallback)
  test_protocol.md          # User testing guide
  maps/                     # Floor plan images (upload via UI)
```

## How the Compiler Works

1. Loads the POI graph from `airport_config.json`
2. For each adjacent POI pair (~59 edges), calls GPT-4o-mini to generate a 1-2 sentence walking direction
3. For every reachable A-to-B pair, BFS finds the shortest path and chains the segment texts
4. All routes are upserted into ChromaDB with metadata for exact matching

This means ~59 LLM calls produce ~464 searchable routes.
