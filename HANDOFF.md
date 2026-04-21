# HANDOFF.md - Airport Wayfinding (2D-to-LLM)

> Last updated: 2026-04-22

## Current State

All 6 build steps complete. The system has two stages: a compiler (Stage 1) that processes floor plan data into searchable route text, and a live interface (Stage 2) that handles voice/text queries via Whisper, intent parsing, ChromaDB retrieval, and TTS.

Graph-based multi-floor routing works. 25 POIs across 3 floors (Underground, Arrivals, Departures) with elevator, escalator, and Skymetro connections. BFS pathfinding verified for cross-floor and cross-terminal routes.

The compiler has NOT been run yet with a live API key. All infrastructure is wired but route generation requires OpenAI API calls (~59 calls for adjacent segments).

## What Was Done This Session

1. Built the full Whisper -> intent -> ChromaDB -> TTS loop with Gradio UI
2. Created manual POI + adjacency JSON for Zurich Airport (25 POIs, 3 floors, 59 edges)
3. Built LLM compiler: generates segment texts for adjacent pairs, BFS chains into ~464 full routes
4. Built VLM extraction module (GPT-4o Vision -> POI + adjacency JSON with comparison against manual)
5. Added multi-floor graph with BFS, fuzzy POI name matching, POI-aware intent parser
6. Created user test protocol (5 tasks, metrics, known limitations)
7. Pushed to GitHub: github.com/terencela/2d-to-llm (public)

## Files Changed

- `app.py` - Gradio UI with Wayfinding + Map Compiler tabs
- `core/pipeline.py`, `core/transcribe.py`, `core/intent.py`, `core/tts.py`, `core/db.py`
- `core/graph.py` - POI graph + BFS + fuzzy matching
- `core/compiler.py` - LLM segment generation + route chaining
- `core/vlm.py` - Vision LLM floor plan extraction
- `data/airport_config.json` - Manual POI + adjacency data
- `data/seed_routes.json` - Hardcoded demo routes (fallback)
- `data/test_protocol.md` - User testing guide

## Open Issues

- Compiler not yet run with live API key (needs OPENAI_API_KEY in .env)
- No real floor plan image tested through VLM extraction
- "H and M" fuzzy match fails (relies on intent parser to normalize)
- No error monitoring, no rate limiting on API calls
- Python project, not TypeScript - different from usual stack

## Exact Next Step

1. Add OPENAI_API_KEY to .env
2. Run `python3 app.py`, go to Map Compiler tab, click "Run Compiler"
3. Verify all 464 routes are generated and queryable
4. Upload a real ZRH floor plan image and test VLM extraction accuracy
5. Test end-to-end with voice input
