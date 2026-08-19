"""Tests for diarization API endpoint integration."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from tests.api.conftest import DUMMY_WAV_HEADER

from insanely_fast_whisper_rocm.core.errors import DiarizationError


@pytest.fixture()
def _mock_diarize_ok() -> Iterator[None]:
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
def _mock_diarize_error() -> Iterator[None]:
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


def test_post_transcriptions__accepts_diarize_form_param(
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


def test_post_transcriptions__includes_speaker_field_in_verbose_json(
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


def test_post_transcriptions__includes_diarized_flag_in_response(
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


def test_post_transcriptions__updates_saved_json_after_post_processing(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Post-processed diarization data is persisted to the saved API JSON file."""
    output_path = tmp_path / "saved-result.json"
    mock_orchestrator.run_transcription.return_value = {
        "text": "Hello world.",
        "chunks": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world.",
            },
        ],
        "output_file_path": str(output_path),
    }
    output_path.write_text(
        json.dumps({"text": "Hello world.", "diarized": False}),
        encoding="utf-8",
    )

    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.diarize",
        side_effect=lambda result, **_: {
            **result,
            "chunks": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world.",
                    "speaker": "SPEAKER_00",
                },
            ],
            "diarized": True,
        },
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            data={
                "diarize": "true",
                "response_format": "verbose_json",
            },
        )

    assert response.status_code == 200
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["diarized"] is True
    assert saved["chunks"][0]["speaker"] == "SPEAKER_00"


def test_post_transcriptions__maps_diarization_error_to_400(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    _mock_diarize_error: None,
) -> None:
    """Any DiarizationError from the route is mapped to HTTP 400."""
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


def test_post_transcriptions__records_diarization_error_in_saved_json(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    tmp_path: Path,
    _mock_diarize_error: None,
) -> None:
    """Diarization failures are recorded on persisted result dicts."""
    output_path = tmp_path / "saved-result.json"
    mock_orchestrator.run_transcription.return_value = {
        "text": "Hello world.",
        "chunks": [{"start": 0.0, "end": 2.0, "text": "Hello world."}],
        "output_file_path": str(output_path),
    }
    output_path.write_text(json.dumps({"text": "Hello world."}), encoding="utf-8")

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
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["diarized"] is False
    assert "HF_TOKEN" in saved["diarization_error"]


@pytest.mark.parametrize(
    ("speaker_options", "error_message"),
    [
        ({"num_speakers": "0"}, "greater than zero"),
        (
            {"num_speakers": "2", "min_speakers": "1"},
            "cannot be combined",
        ),
        (
            {"min_speakers": "3", "max_speakers": "2"},
            "cannot be greater",
        ),
    ],
)
def test_post_transcriptions__rejects_invalid_speaker_config(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
    speaker_options: dict[str, str],
    error_message: str,
) -> None:
    """Invalid speaker-count options return HTTP 400 before diarization runs."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    with patch(
        "insanely_fast_whisper_rocm.core.integrations.diarization.diarize"
    ) as mock_diarize:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            data={
                "diarize": "true",
                "response_format": "json",
                **speaker_options,
            },
        )

    assert response.status_code == 400
    assert error_message in response.json()["detail"]
    mock_diarize.assert_not_called()


def test_post_transcriptions__returns_400_for_invalid_diarization_device(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
) -> None:
    """Invalid diarization_device form value returns an HTTP 400 error."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "diarize": "true",
            "diarization_device": "invalid-device",
            "response_format": "json",
        },
    )
    assert response.status_code == 400
    assert "Invalid diarization_device" in response.json()["detail"]


def test_post_transcriptions__sets_stabilized_flag_on_success(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
) -> None:
    """verbose_json includes stabilized: true when stabilization succeeds."""
    with patch(
        "insanely_fast_whisper_rocm.api.routes.stabilize_timestamps",
        return_value={"text": "Hello", "chunks": []},
    ):
        audio_file = io.BytesIO(DUMMY_WAV_HEADER)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            data={
                "stabilize": "true",
                "response_format": "verbose_json",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body.get("stabilized") is True


def test_stabilized_flag_on_failure(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
) -> None:
    """verbose_json includes stabilized: false when stabilization fails."""
    with patch(
        "insanely_fast_whisper_rocm.api.routes.stabilize_timestamps",
        side_effect=RuntimeError("stabilization crashed"),
    ):
        audio_file = io.BytesIO(DUMMY_WAV_HEADER)
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            data={
                "stabilize": "true",
                "response_format": "verbose_json",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body.get("stabilized") is False


def test_stabilized_flag_absent_when_not_requested(
    client: TestClient,
    mock_orchestrator: pytest.MonkeyPatch,
) -> None:
    """verbose_json omits stabilized key when stabilize is not requested."""
    audio_file = io.BytesIO(DUMMY_WAV_HEADER)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_file, "audio/wav")},
        data={
            "stabilize": "false",
            "response_format": "verbose_json",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "stabilized" not in body
