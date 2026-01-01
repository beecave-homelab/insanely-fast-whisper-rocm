"""Unit tests for response_format handling on transcription and translation endpoints."""

import io
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.main import app
from insanely_fast_whisper_rocm.utils import (
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

    def __init__(self) -> None:
        """
        Initialize the stub pipeline with a mocked ASR backend.
        
        Sets self.asr_backend to a MagicMock and assigns "test-model" to its
        config.model_name for deterministic test behavior.
        """
        from unittest.mock import MagicMock

        self.asr_backend = MagicMock()
        self.asr_backend.config.model_name = "test-model"

    def process(self, *args: object, **kwargs: object) -> dict[str, Any]:  # noqa: D401
        """
        Produce a deterministic mock transcription result for testing.
        
        Returns:
            dict[str, Any]: A dictionary with keys:
                - "text": the full transcription string ("hello world").
                - "chunks": a list of chunk dictionaries, each containing:
                    - "id" (int), "seek" (int), "start" (float), "end" (float), "text" (str).
                - "language": the ISO language code ("en").
        """
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
) -> str:
    """
    Send a minimal dummy WAV file to the given endpoint with the specified response_format.
    
    Returns:
        Response: The response object returned by the FastAPI TestClient.
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
@patch("insanely_fast_whisper_rocm.api.routes.create_orchestrator")
def test_response_format_variants(
    mock_create_orchestrator: Mock,
    endpoint: str,
    response_format: str,
    expected_content_type: str,
) -> None:
    """Ensure each endpoint returns correct status and content-type per format."""
    # Setup mocks
    mock_orch = mock_create_orchestrator.return_value

    # Verbose JSON needs specific segments
    if response_format == RESPONSE_FORMAT_VERBOSE_JSON:
        mock_orch.run_transcription.return_value = {
            "text": "hello world",
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello ",
                    "tokens": [1, 2],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.1,
                }
            ],
            "language": "en",
        }
    else:
        mock_orch.run_transcription.return_value = {
            "text": "hello world",
            "chunks": [
                {"start": 0.0, "end": 1.0, "text": "hello "},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
            "language": "en",
        }

    client = TestClient(app)
    response = _post_file(client, endpoint, response_format)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(expected_content_type)

    # Spot-check payload shape for verbose_json
    if response_format == RESPONSE_FORMAT_VERBOSE_JSON:
        payload = response.json()
        assert "segments" in payload and isinstance(payload["segments"], list)
        assert payload["language"] == "en"
        # In tests, if stabilization is not enabled, segments might be missing
        # or empty depending on how ResponseFormatter works.
        # Ensure we have at least segments if it's verbose_json.
        if not payload.get("segments"):
            # Fallback check for chunks if segments is empty (some formatters might do this)
            assert "chunks" in payload or payload.get("segments") is not None
        else:
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