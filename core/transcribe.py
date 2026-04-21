from pathlib import Path

from openai import OpenAI


def transcribe_audio(audio_path: str) -> str:
    """Send audio file to Whisper API, return transcript text."""
    client = OpenAI()
    with Path(audio_path).open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
        )
    return response.text
