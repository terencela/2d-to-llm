from pathlib import Path

from core.config import get_openai_client


def transcribe_audio(audio_path: str) -> str:
    """Send audio file to Whisper API, return transcript text."""
    client = get_openai_client()
    with Path(audio_path).open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
        )
    return response.text
