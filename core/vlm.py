"""Vision LLM extraction: convert a floor plan image to structured POI + adjacency data."""

import base64
import json
import re
from pathlib import Path

from core.config import get_openai_client, get_logger

log = get_logger("vlm")


EXTRACTION_PROMPT = """You are analyzing a 2D floor plan image of an airport terminal.

Extract ALL visible Points of Interest (POIs) and their spatial relationships.

Return ONLY valid JSON with this exact structure:
{
  "floor_name": "descriptive name for this floor",
  "pois": [
    {
      "id": "lowercase_snake_case_id",
      "name": "Human Readable Name",
      "type": "one of: gate, checkin, shop, restaurant, lounge, service, transport, vertical, hall, security",
      "x": 0.0 to 1.0 (approximate horizontal position, 0=left, 1=right),
      "y": 0.0 to 1.0 (approximate vertical position, 0=top, 1=bottom)
    }
  ],
  "adjacencies": [
    {
      "from": "poi_id_1",
      "to": "poi_id_2",
      "type": "one of: hallway, elevator, escalator, stairs, skymetro, travelator",
      "bidirectional": true or false,
      "notes": "optional: security checkpoint, restricted area, etc."
    }
  ]
}

Rules:
- Include EVERY labeled location, store, gate, counter, elevator, escalator, restroom, and service point
- Two POIs are adjacent if a person can walk directly between them without passing through another labeled POI
- Mark one-way connections (e.g. past security) as bidirectional: false
- Estimate x,y positions relative to the image dimensions
- Use consistent snake_case IDs derived from the name
- Include elevators, escalators, and stairs as POIs with type "vertical"
"""


def _extract_json(text: str) -> dict:
    """Extract JSON from VLM response, handling markdown fences and preamble."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    return json.loads(text)


def encode_image(image_path: str) -> str:
    """Read image file and return base64 string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def extract_pois_from_image(
    image_path: str,
    floor_number: int = 0,
    model: str = "gpt-4o",
) -> dict:
    """Send floor plan image to VLM and extract structured POI + adjacency data."""
    client = get_openai_client()
    b64_image = encode_image(image_path)

    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    log.info("Extracting POIs from %s (floor %d, model %s)", image_path, floor_number, model)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"This is floor {floor_number}. Extract all POIs and adjacencies.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    extracted = _extract_json(content)

    for poi in extracted.get("pois", []):
        poi["floor"] = floor_number

    log.info(
        "Extracted %d POIs, %d adjacencies",
        len(extracted.get("pois", [])),
        len(extracted.get("adjacencies", [])),
    )
    return extracted


def compare_with_manual(
    vlm_data: dict,
    manual_path: str = "data/airport_config.json",
) -> dict:
    """Compare VLM extraction against manual ground truth."""
    manual = json.loads(Path(manual_path).read_text())
    manual_poi_names = {p["name"].lower() for p in manual["pois"]}
    vlm_poi_names = {p["name"].lower() for p in vlm_data.get("pois", [])}

    matched = manual_poi_names & vlm_poi_names
    missed = manual_poi_names - vlm_poi_names
    extra = vlm_poi_names - manual_poi_names

    return {
        "manual_pois": len(manual_poi_names),
        "vlm_pois": len(vlm_poi_names),
        "matched": len(matched),
        "matched_names": sorted(matched),
        "missed_by_vlm": sorted(missed),
        "extra_in_vlm": sorted(extra),
        "manual_adjacencies": len(manual.get("adjacencies", [])),
        "vlm_adjacencies": len(vlm_data.get("adjacencies", [])),
        "poi_recall": round(len(matched) / max(len(manual_poi_names), 1) * 100, 1),
    }


def save_vlm_draft(vlm_data: dict, output_path: str = "data/poi_vlm_draft.json") -> str:
    """Save VLM extraction for human review."""
    Path(output_path).write_text(json.dumps(vlm_data, indent=2))
    return output_path


def extract_and_compare(
    image_path: str,
    floor_number: int = 0,
    manual_path: str = "data/airport_config.json",
) -> tuple[dict, dict]:
    """Full extraction + comparison pipeline. Returns (vlm_data, comparison)."""
    vlm_data = extract_pois_from_image(image_path, floor_number)
    save_vlm_draft(vlm_data)
    comparison = compare_with_manual(vlm_data, manual_path)
    return vlm_data, comparison
