import tempfile
from pathlib import Path

from openai import OpenAI


def text_to_speech(text: str) -> str:
    """Convert text to speech via OpenAI TTS. Returns path to mp3 file."""
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    response.stream_to_file(Path(tmp.name))
    return tmp.name
