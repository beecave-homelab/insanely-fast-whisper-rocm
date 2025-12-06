"""Unit tests for response_format handling on transcription and translation endpoints."""

import io
<<<<<<< HEAD:tests/test_response_formats.py
=======
from collections.abc import Generator
>>>>>>> dev:tests/api/test_response_formats.py
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

<<<<<<< HEAD:tests/test_response_formats.py
from insanely_fast_whisper_api.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_api.main import app
from insanely_fast_whisper_api.utils import (
=======
from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.main import app
from insanely_fast_whisper_rocm.utils import (
>>>>>>> dev:tests/api/test_response_formats.py
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_VTT,
)

# ---------------------------------------------------------------------------
# Dummy dependencies --------------------------------------------------------
# ---------------------------------------------------------------------------


class _StubPipeline:  # noqa: D401 (simple class)
    """Minimal stub for WhisperPipeline that returns deterministic output."""

<<<<<<< HEAD:tests/test_response_formats.py
    def process(self, *_, **__) -> dict[str, Any]:  # noqa: D401
=======
    def process(self, *args: object, **kwargs: object) -> dict[str, Any]:  # noqa: D401
>>>>>>> dev:tests/api/test_response_formats.py
        return {
            "text": "hello world",
            "chunks": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello ",
                },
                {
                    "id": 1,
                    "seek": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "world",
                },
            ],
            "language": "en",
        }


class _StubFileHandler:  # noqa: D401
    """Stub for FileHandler that bypasses filesystem interaction."""

    def validate_audio_file(self, *args: object) -> bool:
        return True

    def save_upload(self, _file: object) -> str:  # noqa: D401
        return "dummy_path.wav"

    def cleanup(self, _: object) -> bool:
        return True


@pytest.fixture(autouse=True)
def _override_dependencies() -> Generator[None, None, None]:
    """Override FastAPI dependencies for tests using app.dependency_overrides."""

    def _get_stub_asr_pipeline() -> "_StubPipeline":  # type: ignore[return-value]
        return _StubPipeline()

    def _get_stub_file_handler() -> "_StubFileHandler":  # type: ignore[return-value]
        return _StubFileHandler()

    app.dependency_overrides[get_asr_pipeline] = _get_stub_asr_pipeline
    app.dependency_overrides[get_file_handler] = _get_stub_file_handler

    yield

    app.dependency_overrides = {}


def _post_file(
    client: TestClient, url: str, response_format: str = RESPONSE_FORMAT_JSON
) -> requests.Response:
    """Helper to send a dummy wav file to a URL with specified response_format.

    Returns:
        requests.Response: Response object from the FastAPI TestClient.
    """
    dummy_audio = io.BytesIO(
        b"RIFF\x00\x00\x00\x00WAVEfmt "
    )  # Minimal WAV header bytes
    files = {"file": ("test.wav", dummy_audio, "audio/wav")}
    data = {"response_format": response_format}
    return client.post(url, files=files, data=data)


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    ],
)
@pytest.mark.parametrize(
    "response_format,expected_content_type",
    [
        (RESPONSE_FORMAT_JSON, "application/json"),
        (RESPONSE_FORMAT_VERBOSE_JSON, "application/json"),
        (RESPONSE_FORMAT_TEXT, "text/plain; charset=utf-8"),
        (RESPONSE_FORMAT_SRT, "text/srt; charset=utf-8"),
        (RESPONSE_FORMAT_VTT, "text/vtt; charset=utf-8"),
    ],
)
def test_response_format_variants(
    endpoint: str, response_format: str, expected_content_type: str
) -> None:
    """Ensure each endpoint returns correct status and content-type per format."""
    client = TestClient(app)
    response = _post_file(client, endpoint, response_format)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(expected_content_type)

    # Spot-check payload shape for verbose_json
    if response_format == RESPONSE_FORMAT_VERBOSE_JSON:
        payload = response.json()
        assert "segments" in payload and isinstance(payload["segments"], list)
        assert payload["language"] == "en"
        # Ensure required keys present in first segment
        seg_keys = {
            "id",
            "seek",
            "start",
            "end",
            "text",
            "tokens",
            "temperature",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
        }
        assert seg_keys <= payload["segments"][0].keys()

    if response_format == RESPONSE_FORMAT_TEXT:
        assert response.text == "hello world"
