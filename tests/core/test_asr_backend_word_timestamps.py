"""Tests for word-level timestamp handling in the ASR backend.

This test suite verifies that word-level timestamps work correctly,
particularly ensuring that chunk_length_s is disabled when using
return_timestamps="word" to avoid Transformers pipeline conflicts.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)


def test_disables_chunk_length_for_word_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure chunk_length_s is disabled when return_timestamps='word'.

    This test verifies the fix for the word-level timestamp bug where
    using both chunk_length_s and return_timestamps="word" causes
    Transformers to assign all words the same timestamp (chunk boundary).

    The fix should set chunk_length_s to None when word-level timestamps
    are requested, allowing the manual chunking in pipeline.py to handle
    audio splitting while Transformers processes each chunk without
    further internal chunking.

    Args:
        monkeypatch: Pytest fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-base",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    # Track the kwargs passed to the pipeline
    captured_kwargs: dict[str, Any] = {}

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,  # Valid timestamp token
            task_to_id={"transcribe": 50359, "translate": 50358},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Capture kwargs and return mock word-level data.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result with word-level timestamps.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()

            # Return mock word-level timestamp data
            return {
                "text": "Hello world",
                "chunks": [
                    {"text": "Hello", "timestamp": [0.0, 0.5]},
                    {"text": "world", "timestamp": [0.5, 1.0]},
                ],
            }

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    # Request word-level timestamps
    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value="word",
    )

    # CRITICAL ASSERTION: chunk_length_s should be None when word timestamps requested
    assert captured_kwargs.get("chunk_length_s") is None, (
        "chunk_length_s must be None for word-level timestamps to avoid Transformers bug"
    )

    # Verify return_timestamps is set to "word"
    assert captured_kwargs.get("return_timestamps") == "word"

    # Verify the result contains word-level data
    assert result["text"] == "Hello world"
    assert result["chunks"] is not None
    assert len(result["chunks"]) == 2


def test_keeps_chunk_length_for_chunk_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Ensure chunk_length_s is preserved when return_timestamps=True (chunk-level).

    This test verifies that chunk-level timestamps still use chunk_length_s
    as before, since the bug only affects word-level timestamps.

    Args:
        monkeypatch: Pytest fixture for mocking.
        tmp_path: Pytest fixture for temporary directory.
    """
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-base",
        device="cpu",
        dtype="float32",
        batch_size=4,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    captured_kwargs: dict[str, Any] = {}

    class DummyModel:
        generation_config = types.SimpleNamespace(
            no_timestamps_token_id=50363,
            task_to_id={"transcribe": 50359},
            lang_to_id={"en": 50259},
        )
        config = types.SimpleNamespace(
            lang_to_id={"en": 50259}, task_to_id={"transcribe": 50359}
        )

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            """Mock pipeline call.

            Args:
                path: Audio file path.
                **kwargs: Pipeline keyword arguments.

            Returns:
                Mock transcription result.
            """
            nonlocal captured_kwargs
            captured_kwargs = kwargs.copy()
            return {
                "text": "Hello world",
                "chunks": [{"text": "Hello world", "timestamp": [0.0, 2.0]}],
            }

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    # Request chunk-level timestamps (return_timestamps=True)
    result = backend.process_audio(
        str(audio_file),
        language=None,
        task="transcribe",
        return_timestamps_value=True,
    )

    # For chunk-level timestamps, chunk_length_s should still be set
    assert captured_kwargs.get("chunk_length_s") == 30
    assert captured_kwargs.get("return_timestamps") is True
    assert result["text"] == "Hello world"
