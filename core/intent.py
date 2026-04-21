import json

from openai import OpenAI


SYSTEM_PROMPT = """You extract navigation intent from user queries at an airport.
Return ONLY valid JSON with two keys: "start" and "end".
Both values are point-of-interest names exactly as they appear in the known locations list.

If the user doesn't mention a starting location, set "start" to "unknown".
If the user doesn't mention a destination, set "end" to "unknown".

Match user input to the closest known location name. For example:
- "H and M" or "HM" -> "H&M"
- "train station" or "SBB" -> "SBB Train Station"
- "check in" or "checkin 2" -> "Check-in 2"
- "gate A 15" -> "Gate A15"
- "baggage" or "luggage" -> "Baggage Claim"

Known locations:
{poi_names}
"""


_poi_names_cache: list[str] | None = None


def set_known_pois(names: list[str]) -> None:
    """Update the list of known POI names for intent matching."""
    global _poi_names_cache
    _poi_names_cache = names


def parse_intent(text: str) -> dict[str, str]:
    """Extract start and end POIs from a natural language query."""
    poi_list = "\n".join(f"- {n}" for n in (_poi_names_cache or []))
    prompt = SYSTEM_PROMPT.format(poi_names=poi_list if poi_list else "No locations loaded yet.")

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return {
        "start": parsed.get("start", "unknown"),
        "end": parsed.get("end", "unknown"),
    }
