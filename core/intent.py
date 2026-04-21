import json

from openai import OpenAI


SYSTEM_PROMPT = """You extract navigation intent from user queries at an airport.
Return ONLY valid JSON with two keys: "start" and "end".
Both values are point-of-interest names exactly as a user would say them.

If the user doesn't mention a starting location, set "start" to "unknown".
If the user doesn't mention a destination, set "end" to "unknown".

Examples:
- "How do I get from H&M to Check-in 2?" -> {"start": "H&M", "end": "Check-in 2"}
- "I'm at Gate A15, where is the lounge?" -> {"start": "Gate A15", "end": "Lounge A"}
- "Take me to baggage claim from Gate B32" -> {"start": "Gate B32", "end": "Baggage Claim"}
"""


def parse_intent(text: str) -> dict[str, str]:
    """Extract start and end POIs from a natural language query."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return {
        "start": parsed.get("start", "unknown"),
        "end": parsed.get("end", "unknown"),
    }
