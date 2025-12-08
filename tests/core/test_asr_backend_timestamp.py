"""Tests for timestamp behavior in the ASR backend."""

from __future__ import annotations

import pathlib
import types

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_disables_timestamps_when_generation_config_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Disable timestamps when generation_config lacks no_timestamps_token_id."""
    config = HuggingFaceBackendConfig(
        model_name="dummy-model",
        device="cpu",
        dtype="float32",
        batch_size=1,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(no_timestamps_token_id=None)
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: object) -> dict[str, str]:  # noqa: ARG002
            mapping = kwargs if isinstance(kwargs, dict) else {}
            assert mapping.get("return_timestamps") is False
            return {"text": "hello"}

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file), language=None, task="transcribe", return_timestamps_value=True
    )

    assert result["chunks"] is None
