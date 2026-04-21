# Airport Wayfinding (2D-to-LLM)

Voice and text-based wayfinding for airports. Upload a 2D floor plan, extract POIs via vision LLM, generate route descriptions, and serve them through a voice interface.

## Current State: Stage 1 (Hardcoded Routes)

The retrieval loop is wired: Whisper transcription, GPT-4o-mini intent parsing, ChromaDB retrieval, and OpenAI TTS output. 8 demo routes for Zurich Airport are pre-seeded.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Run

```bash
python app.py
```

Opens a Gradio UI at `http://localhost:7860`. Speak into the mic or type a question like "How do I get from H&M to Check-in 2?"

## Architecture

```
Voice/Text -> Whisper STT -> Intent Parser (GPT-4o-mini) -> ChromaDB Retrieval -> TTS -> Audio
```

## Project Structure

```
app.py              # Gradio UI
core/
  pipeline.py       # Orchestrates the full flow
  transcribe.py     # Whisper STT
  intent.py         # Extract start/end POI
  tts.py            # OpenAI TTS
  db.py             # ChromaDB init, seed, query
data/
  seed_routes.json  # Hardcoded demo routes
```
