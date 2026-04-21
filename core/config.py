"""Shared configuration: OpenAI client, logging, temp file management."""

import logging
import tempfile
from pathlib import Path

from openai import OpenAI

_openai_client: OpenAI | None = None
_temp_files: list[str] = []


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"wayfinding.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def create_temp_file(suffix: str = ".mp3") -> str:
    """Create a tracked temp file. Call cleanup_temp_files() to remove all."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    _temp_files.append(tmp.name)
    return tmp.name


def cleanup_temp_files() -> int:
    """Remove all tracked temp files. Returns count removed."""
    removed = 0
    for path in _temp_files:
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    _temp_files.clear()
    return removed
