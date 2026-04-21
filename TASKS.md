# TASKS.md - Airport Wayfinding (2D-to-LLM)

> Track what's done and what's next.

## Done

- [x] Gradio UI with voice + text input
- [x] Whisper STT integration
- [x] GPT-4o-mini intent parser (POI-aware)
- [x] ChromaDB setup, seeding, metadata + semantic query
- [x] OpenAI TTS output (Nova voice)
- [x] Manual POI + adjacency JSON for ZRH (25 POIs, 3 floors)
- [x] Graph module with BFS pathfinding
- [x] Fuzzy POI name matching
- [x] LLM compiler (adjacent pair segments + BFS chaining)
- [x] VLM extraction module (GPT-4o Vision)
- [x] VLM vs manual comparison tool
- [x] Multi-floor support (elevator, escalator, Skymetro)
- [x] Map Compiler tab in Gradio UI
- [x] User test protocol
- [x] GitHub repo setup + pushed

## Pending

- [ ] Run compiler with live API key and verify all routes
- [ ] Test VLM extraction with real ZRH floor plan
- [ ] End-to-end voice test
- [ ] Add POI aliases (alternative names users might say)
- [ ] Add German language support (ZRH is bilingual)
- [ ] Add "I don't know where I am" fallback flow
- [ ] Error monitoring / logging
- [ ] Performance optimization (batch LLM calls in compiler)
- [ ] Deploy to a shareable URL for airport testing
