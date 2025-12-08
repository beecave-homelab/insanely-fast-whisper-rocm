"""Integration tests for the WebUI using gradio_client.Client.

These tests rely on the `webui_server` fixture defined in ``tests/conftest.py`` which starts
up the WebUI once per test session.

Mark them with ``pytest.mark.webui`` so they can be skipped in CI when desired.
"""

from pathlib import Path
from typing import Any

import pytest
from gradio_client import Client, handle_file

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR.parent / "data"
AUDIO_FILE = DATA_DIR / "conversion-test-file.mp3"
LONG_AUDIO_FILE = DATA_DIR / "test.mp3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.webui]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _predict(client: Client, audio_path: Path) -> list[Any]:
    """Helper that calls predict on the /transcribe_audio_v2 endpoint.

    Returns:
        list[Any]: Response list where index 0 is the transcription string,
        index 1 is a human-readable processing time, index 2 is the config
        dictionary, followed by optional download URLs.
    """
    return client.predict(
        handle_file(audio_path),  # audio_input (List[str]) via wrapper
        "openai/whisper-tiny",  # model
        "cpu",  # device
        16,  # batch_size
        "chunk",  # timestamp_type
        "en",  # language
        "transcribe",  # task
        "float16",  # dtype
        30,  # chunk_length
        True,  # save_transcriptions
        "transcripts",  # temp_uploads_dir
        api_name="/transcribe_audio_v2",
    )


def test_ui_root_accessible(webui_server: str) -> None:
    """Ensure the root URL responds and Gradio app is present."""
    import requests

    html = requests.get(webui_server, timeout=10).text.lower()
    assert "<gradio-app" in html, "gradio-app container not found"


@pytest.mark.skipif(not AUDIO_FILE.exists(), reason="Sample mp3 missing")
def test_short_audio_transcription(webui_server: str) -> None:
    """Transcribe a short audio file and verify core response fields."""
    client = Client(webui_server)
    result = _predict(client, AUDIO_FILE)

    # result is a list: [transcription:str, summary/info:str, config:dict, *download_urls]
    assert isinstance(result, list) and len(result) >= 3, "Unexpected response format"

    transcription, processing_time, config_used = result[:3]

    assert isinstance(transcription, str) and transcription, "Empty transcription"
    assert "second" in str(processing_time).lower(), "Processing time missing 'second'"
    assert isinstance(config_used, dict) and config_used.get("model"), (
        "Config missing model"
    )


@pytest.mark.skipif(not LONG_AUDIO_FILE.exists(), reason="Long test audio missing")
def test_long_audio_transcription(webui_server: str) -> None:
    """Transcribe a longer file; mainly a runtime smoke-test."""
    client = Client(webui_server)
    result = _predict(client, LONG_AUDIO_FILE)
    transcription = result[0]
    assert isinstance(transcription, str) and len(transcription) > 0
