"""Zurich Airport Wayfinding - Voice & Text interface."""

import gradio as gr
from dotenv import load_dotenv

from core.db import seed_from_json, get_collection
from core.pipeline import run_voice, run_text

load_dotenv()

AVAILABLE_ROUTES = [
    "H&M -> Check-in 2",
    "Check-in 2 -> H&M",
    "Check-in 2 -> Gate A15",
    "Migros -> SBB Train Station",
    "SBB Train Station -> Check-in 2",
    "Gate A15 -> Lounge A",
    "Lounge A -> Gate B32",
    "Gate B32 -> Baggage Claim",
]


def ensure_db_seeded() -> None:
    collection = get_collection()
    if collection.count() == 0:
        count = seed_from_json()
        print(f"Seeded {count} routes into ChromaDB")


def handle_voice(audio_path: str | None):
    if audio_path is None:
        return "", "", "", "", None
    result = run_voice(audio_path)
    return (
        result.transcript,
        result.start_poi,
        result.end_poi,
        result.route_text,
        result.audio_path,
    )


def handle_text(text: str):
    if not text.strip():
        return "", "", "", None
    result = run_text(text)
    return (
        result.start_poi,
        result.end_poi,
        result.route_text,
        result.audio_path,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Airport Wayfinding",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Airport Wayfinding\n"
            "Ask for directions between locations at Zurich Airport. "
            "Speak into the microphone or type your question."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Voice Input")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your question",
                )
                voice_btn = gr.Button("Get Directions (Voice)", variant="primary")

                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    label="Or type your question",
                    placeholder="How do I get from H&M to Check-in 2?",
                    lines=2,
                )
                text_btn = gr.Button("Get Directions (Text)", variant="primary")

                gr.Markdown("### Available Routes (Step 1 Demo)")
                gr.Markdown("\n".join(f"- {r}" for r in AVAILABLE_ROUTES))

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                transcript_out = gr.Textbox(label="Transcription", interactive=False)
                start_out = gr.Textbox(label="Start POI", interactive=False)
                end_out = gr.Textbox(label="Destination POI", interactive=False)
                route_out = gr.Textbox(
                    label="Directions", interactive=False, lines=4
                )
                audio_out = gr.Audio(label="Audio Directions", type="filepath")

        voice_btn.click(
            fn=handle_voice,
            inputs=[audio_input],
            outputs=[transcript_out, start_out, end_out, route_out, audio_out],
        )

        text_btn.click(
            fn=handle_text,
            inputs=[text_input],
            outputs=[start_out, end_out, route_out, audio_out],
        )

    return demo


if __name__ == "__main__":
    ensure_db_seeded()
    demo = build_ui()
    demo.launch()
