"""Tests for backend generation config behavior."""

from __future__ import annotations

import pathlib
import types

import pytest

from insanely_fast_whisper_api.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_omits_task_when_generation_config_outdated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure task/language are omitted when generation_config lacks mappings."""
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
        generation_config = types.SimpleNamespace(task_to_id=None, lang_to_id=None)
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: object) -> dict[str, str]:  # noqa: ARG002
            gen_kwargs = {}
            if isinstance(kwargs, dict):
                gen_kwargs = dict(kwargs).get("generate_kwargs", {})
            assert "task" not in gen_kwargs
            assert "language" not in gen_kwargs
            return {"text": "hi"}

    backend.asr_pipe = DummyPipe()

    monkeypatch.setattr(
        "insanely_fast_whisper_api.audio.conversion.ensure_wav",
        lambda p: pathlib.Path(p),
    )

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    result = backend.process_audio(
        str(audio_file), language=None, task="translate", return_timestamps_value=False
    )

    assert result["text"] == "hi"
