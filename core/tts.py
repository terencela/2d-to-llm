from pathlib import Path

from core.config import get_openai_client, create_temp_file


def text_to_speech(text: str) -> str:
    """Convert text to speech via OpenAI TTS. Returns path to mp3 file."""
    client = get_openai_client()
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    tmp_path = create_temp_file(suffix=".mp3")
    response.stream_to_file(Path(tmp_path))
    return tmp_path
