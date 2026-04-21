# DECISIONS.md - Airport Wayfinding (2D-to-LLM)

> Why things are the way they are.

## 2026-04-22: Python + Gradio (not Next.js)

**Decision**: Python stack with Gradio, not the usual Next.js + TypeScript.

**Why**: This is an AI pipeline project, not a web product. Every dependency (Whisper, ChromaDB, OpenAI SDK, VLM) has a Python-first SDK. Gradio gives a functional UI for prototyping without frontend build complexity. If this becomes a production product, the backend stays Python and gets a proper frontend later.

## 2026-04-22: ChromaDB (not Pinecone, not pgvector)

**Decision**: ChromaDB with local persistent storage.

**Why**: Runs locally with zero config. No cloud dependency, no API key needed for the vector DB itself. Good enough for thousands of routes. Metadata filtering + semantic search cover both exact and fuzzy retrieval. Can migrate to pgvector/Pinecone later if scale demands it.

## 2026-04-22: Adjacent-pair-only LLM generation

**Decision**: Only call the LLM for directly adjacent POI pairs (~59), not all pairs (~464).

**Why**: Saves 87% of LLM calls. Non-adjacent routes are built by BFS pathfinding + programmatic chaining of segment texts. Quality is comparable because each segment is independently generated with proper context.

## 2026-04-22: Metadata-first retrieval with semantic fallback

**Decision**: Query ChromaDB by metadata filter (exact start_poi + end_poi match) first, then fall back to semantic search.

**Why**: Pre-compiled routes have exact metadata. Semantic search is the safety net for queries the intent parser slightly misinterprets. This gives deterministic results for known pairs and graceful degradation for edge cases.

## 2026-04-22: Explicit graph nodes per floor for vertical connectors

**Decision**: Elevators and escalators are separate POI nodes per floor (e.g., elevator_central_0, elevator_central_1, elevator_central_2) with adjacency edges between them.

**Why**: Makes BFS work without special-casing floor transitions. The graph naturally routes through floor changes. Downside: more nodes, but with 25 total POIs this is negligible.
