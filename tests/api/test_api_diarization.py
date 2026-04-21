"""Tests for diarization API endpoint integration."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import DUMMY_WAV_HEADER

from insanely_fast_whisper_rocm.core.errors import DiarizationError


@pytest.fixture()
def _mock_diarize_ok() -> None:
    """Mock diarize to return a successful diarized result.

    Yields:
        None: Control returns to the test after the patch is applied.
    """
    diarized_result = {
        "text": "Hello world.",
        "chunks": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world.",
                "speaker": "SPEAKER_00",
            },
        ],
        "diarized": True,
    }
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.diarize",
        return_value=diarized_result,
    ):
        yield


@pytest.fixture()
def _mock_diarize_error() -> None:
    """Mock diarize to raise DiarizationError.

    Yields:
        None: Control returns to the test after the patch is applied.
    """
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.diarize",
        side_effect=DiarizationError(
            "HF_TOKEN required",
            model="pyannote/speaker-diarization-3.1",
            reason="missing_token",
        ),
    ):
        yield


def test_diarize_form_param_accepted(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    _mock_diarize_ok: None,
) -> None:
    """diarize=True form param is accepted and routed correctly."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "response_format": "verbose_json",
        },
    )
    assert response.status_code == 200


def test_speaker_field_in_verbose_json(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    _mock_diarize_ok: None,
) -> None:
    """verbose_json response includes speaker field when diarized."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "response_format": "verbose_json",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "segments" in body
    assert any("speaker" in seg for seg in body["segments"])


def test_diarized_flag_in_response(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    _mock_diarize_ok: None,
) -> None:
    """verbose_json response includes diarized: true when diarization applied."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "response_format": "verbose_json",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("diarized") is True


def test_error_response_when_hf_token_missing(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    _mock_diarize_error: None,
) -> None:
    """400 error when diarize=True but HF_TOKEN not set."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "response_format": "json",
        },
    )
    assert response.status_code == 400
    assert "HF_TOKEN" in response.json()["detail"]
