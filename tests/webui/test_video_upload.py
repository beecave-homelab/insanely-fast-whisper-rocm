"""Video upload integration test using gradio_client.Client."""

from pathlib import Path

import pytest
from gradio_client import Client, handle_file

pytestmark = [pytest.mark.webui]

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR.parent / "data"
VIDEO_FILE = DATA_DIR / "sample.mp4"


@pytest.mark.skipif(not VIDEO_FILE.exists(), reason="Sample mp4 missing")
def test_video_upload_transcription(webui_server):
    """Upload a small MP4 and assert transcription not empty."""
    client = Client(webui_server)
    result = client.predict(
        handle_file(VIDEO_FILE),
        "openai/whisper-tiny",
        "cpu",
        16,
        "chunk",
        "en",
        "transcribe",
        "float16",
        30,
        True,
        "transcripts",
        api_name="/transcribe_audio_v2",
    )

    transcription = result[0]
    assert isinstance(transcription, str) and transcription.strip(), "Empty transcription from video"
