"""Zurich Airport Wayfinding - Voice, Text, and Map Compiler interface."""

import atexit
import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from core.config import get_logger, cleanup_temp_files
from core.db import seed_from_json, get_collection, get_route_count, reset_collection
from core.graph import load_graph
from core.intent import set_known_pois
from core.pipeline import run_voice, run_text, set_graph

load_dotenv()

log = get_logger("app")
_current_graph = None


def init_app() -> None:
    """Load graph, seed POI names into intent parser, seed DB if empty."""
    global _current_graph
    config_path = "data/airport_config.json"

    if Path(config_path).exists():
        _current_graph = load_graph(config_path)
        set_graph(_current_graph)
        set_known_pois(_current_graph.get_all_poi_names())

    collection = get_collection()
    if collection.count() == 0:
        if Path("data/compiled_routes.json").exists():
            count = _seed_compiled_routes()
            log.info("Seeded %d compiled routes", count)
        elif Path("data/seed_routes.json").exists():
            count = seed_from_json()
            log.info("Seeded %d demo routes", count)


def _seed_compiled_routes() -> int:
    """Seed ChromaDB from previously compiled routes."""
    data = json.loads(Path("data/compiled_routes.json").read_text())
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for route in data:
        route_id = f"{route['start']}|{route['end']}"
        ids.append(route_id)
        documents.append(route["route_text"])
        metadatas.append({
            "start_poi": route["start"],
            "end_poi": route["end"],
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def handle_voice(audio_path: str | None):
    if audio_path is None:
        return "", "", "", "", "", None
    result = run_voice(audio_path)
    return (
        result.transcript,
        result.start_poi,
        result.end_poi,
        result.floor_info,
        result.route_text,
        result.audio_path,
    )


def handle_text(text: str):
    if not text.strip():
        return "", "", "", "", None
    result = run_text(text)
    return (
        result.start_poi,
        result.end_poi,
        result.floor_info,
        result.route_text,
        result.audio_path,
    )


def handle_compile():
    """Run the full compiler: generate segments via LLM, chain routes, push to ChromaDB."""
    from core.compiler import run_compiler

    try:
        reset_collection()
        stats = run_compiler()
        return (
            f"Compilation complete.\n"
            f"POIs: {stats['pois']}\n"
            f"Edges: {stats['edges']}\n"
            f"Segments generated: {stats['segments']}\n"
            f"Full routes built: {stats['routes']}\n"
            f"Routes in ChromaDB: {stats['stored']}"
        )
    except Exception as e:
        return f"Compilation failed: {e}"


def handle_vlm_extract(image, floor_number):
    """Extract POIs from uploaded floor plan via VLM."""
    from core.vlm import extract_and_compare

    if image is None:
        return "No image uploaded.", ""

    try:
        vlm_data, comparison = extract_and_compare(
            image_path=image,
            floor_number=int(floor_number),
        )

        result_lines = [
            f"Extracted {len(vlm_data.get('pois', []))} POIs, "
            f"{len(vlm_data.get('adjacencies', []))} adjacencies",
            "",
            f"POI Recall vs manual: {comparison['poi_recall']}%",
            f"Matched: {comparison['matched']} / {comparison['manual_pois']}",
        ]

        if comparison["missed_by_vlm"]:
            result_lines.append(f"Missed: {', '.join(comparison['missed_by_vlm'])}")
        if comparison["extra_in_vlm"]:
            result_lines.append(f"Extra: {', '.join(comparison['extra_in_vlm'])}")

        result_lines.append("\nVLM draft saved to data/poi_vlm_draft.json")

        vlm_json = json.dumps(vlm_data, indent=2)
        return "\n".join(result_lines), vlm_json

    except Exception as e:
        return f"Extraction failed: {e}", ""


def get_poi_list():
    """Return formatted list of known POIs for display."""
    if _current_graph is None:
        return "No airport config loaded."

    floors: dict[int, list[str]] = {}
    for poi in _current_graph.pois.values():
        if poi.poi_type == "vertical":
            continue
        floors.setdefault(poi.floor, []).append(poi.name)

    lines = []
    config = json.loads(Path("data/airport_config.json").read_text())
    floor_names = {f["id"]: f["name"] for f in config["floors"]}

    for floor_id in sorted(floors.keys()):
        floor_name = floor_names.get(floor_id, f"Floor {floor_id}")
        lines.append(f"**{floor_name}**")
        for name in sorted(floors[floor_id]):
            lines.append(f"- {name}")
        lines.append("")

    route_count = get_route_count()
    lines.append(f"*{route_count} routes in database*")

    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Airport Wayfinding",
    ) as demo:
        gr.Markdown("# Airport Wayfinding System")

        with gr.Tabs():
            with gr.TabItem("Wayfinding"):
                gr.Markdown(
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

                        gr.Markdown("### Known Locations")
                        poi_display = gr.Markdown(get_poi_list())

                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        transcript_out = gr.Textbox(label="Transcription", interactive=False)
                        start_out = gr.Textbox(label="Start POI", interactive=False)
                        end_out = gr.Textbox(label="Destination POI", interactive=False)
                        floor_out = gr.Textbox(label="Floor Info", interactive=False)
                        route_out = gr.Textbox(label="Directions", interactive=False, lines=5)
                        audio_out = gr.Audio(label="Audio Directions", type="filepath")

                voice_btn.click(
                    fn=handle_voice,
                    inputs=[audio_input],
                    outputs=[transcript_out, start_out, end_out, floor_out, route_out, audio_out],
                )

                text_btn.click(
                    fn=handle_text,
                    inputs=[text_input],
                    outputs=[start_out, end_out, floor_out, route_out, audio_out],
                )

            with gr.TabItem("Map Compiler"):
                gr.Markdown(
                    "### Compile Routes\n"
                    "Generate walking directions for all POI pairs from the airport config. "
                    "Requires an OpenAI API key. Each adjacent pair triggers one LLM call."
                )

                compile_btn = gr.Button("Run Compiler", variant="primary")
                compile_output = gr.Textbox(
                    label="Compiler Output", interactive=False, lines=8
                )

                compile_btn.click(
                    fn=handle_compile,
                    inputs=[],
                    outputs=[compile_output],
                )

                gr.Markdown("---")

                gr.Markdown(
                    "### VLM Floor Plan Extraction\n"
                    "Upload a floor plan image to extract POIs automatically. "
                    "The result is compared against the manual airport config."
                )

                with gr.Row():
                    with gr.Column():
                        floor_image = gr.Image(
                            label="Upload Floor Plan",
                            type="filepath",
                        )
                        floor_number = gr.Number(
                            label="Floor Number",
                            value=0,
                            precision=0,
                        )
                        extract_btn = gr.Button("Extract POIs", variant="secondary")

                    with gr.Column():
                        extract_result = gr.Textbox(
                            label="Extraction Result", interactive=False, lines=8
                        )
                        extract_json = gr.Textbox(
                            label="Extracted JSON", interactive=False, lines=12
                        )

                extract_btn.click(
                    fn=handle_vlm_extract,
                    inputs=[floor_image, floor_number],
                    outputs=[extract_result, extract_json],
                )

    return demo


if __name__ == "__main__":
    atexit.register(cleanup_temp_files)
    init_app()
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())
