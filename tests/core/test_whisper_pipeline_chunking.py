"""Tests for pipeline-driven chunking in `WhisperPipeline`."""

from __future__ import annotations

import types
from typing import Any

import pytest

from insanely_fast_whisper_rocm.core.asr_backend import ASRBackend
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.core.progress import ProgressCallback


class _RecordingBackend(ASRBackend):
    """Backend stub that records invocations and returns queued responses."""

    def __init__(self, responses: list[dict[str, Any]], chunk_length: int = 30) -> None:
        """Initialize the stub backend.

        Args:
            responses: Ordered list of canned responses returned per invocation.
            chunk_length: Exposed chunk length via ``config``.
        """
        self.config = types.SimpleNamespace(chunk_length=chunk_length)
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def process_audio(  # type: ignore[override]
        self,
        audio_file_path: str,
        language: str | None,
        task: str,
        return_timestamps_value: bool | str,
        progress_cb: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Record invocation details and return the next queued response.

        Args:
            audio_file_path: Audio chunk path provided by the pipeline.
            language: Optional language code requested by the pipeline.
            task: Requested Whisper task (``transcribe`` or ``translate``).
            return_timestamps_value: Timestamp flag forwarded by the pipeline.
            progress_cb: Progress callback proxy supplied by the pipeline.
            cancellation_token: Cooperative cancellation token.

        Returns:
            dict[str, Any]: Copy of the canned response for deterministic tests.

        Raises:
            AssertionError: If invoked more times than responses provided.
        """
        self.calls.append({
            "path": audio_file_path,
            "language": language,
            "task": task,
            "return_timestamps": return_timestamps_value,
            "progress_cb_has_on_completed": hasattr(progress_cb, "on_completed"),
        })
        try:
            response = self._responses.pop(0)
        except IndexError as exc:  # pragma: no cover - defensive
            raise AssertionError("Backend received more calls than expected") from exc
        return dict(response)


class _ProgressRecorder:
    """Progress callback implementation capturing pipeline phases."""

    def __init__(self) -> None:
        """Initialize storage for captured progress events."""
        self.audio_started: list[str] = []
        self.audio_finished: list[float | None] = []
        self.chunking_started: list[int | None] = []
        self.chunk_done: list[int] = []
        self.inference_started: list[int | None] = []
        self.inference_batch_done: list[int] = []
        self.completed_count = 0
        self.error_messages: list[str] = []

    def on_model_load_started(self) -> None:
        """Capture model-load start notifications (unused in assertions)."""
        return

    def on_model_load_finished(self) -> None:
        """Capture model-load completion notifications (unused in assertions)."""
        return

    def on_audio_loading_started(self, path: str) -> None:
        """Record the path passed to `on_audio_loading_started`.

        Args:
            path: Path of the audio file being processed.
        """
        self.audio_started.append(path)

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:
        """Record completion of audio loading.

        Args:
            duration_sec: Optional duration reported by the pipeline.
        """
        self.audio_finished.append(duration_sec)

    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Record the total chunk count reported by the pipeline.

        Args:
            total_chunks: Number of chunks to be processed.
        """
        self.chunking_started.append(total_chunks)

    def on_chunk_done(self, index: int) -> None:
        """Record a completed chunk index.

        Args:
            index: Zero-based index for the processed chunk.
        """
        self.chunk_done.append(index)

    def on_inference_started(self, total_batches: int | None) -> None:
        """Record the batch count supplied to `on_inference_started`.

        Args:
            total_batches: Total number of batches to be processed.
        """
        self.inference_started.append(total_batches)

    def on_inference_batch_done(self, index: int) -> None:
        """Record an inference batch completion event.

        Args:
            index: Zero-based index of the completed batch.
        """
        self.inference_batch_done.append(index)

    def on_postprocess_started(self, name: str) -> None:
        """Capture post-processing start events (unused in assertions).

        Args:
            name: Identifier for the post-processing step.
        """
        return

    def on_postprocess_finished(self, name: str) -> None:
        """Capture post-processing completion events (unused in assertions).

        Args:
            name: Identifier for the post-processing step.
        """
        return

    def on_export_started(self, total_items: int) -> None:
        """Capture export start events (unused in assertions).

        Args:
            total_items: Number of items that will be exported.
        """
        return

    def on_export_item_done(self, index: int, label: str) -> None:
        """Capture export item completion events (unused in assertions).

        Args:
            index: Zero-based index of the exported item.
            label: Descriptive label for the exported item.
        """
        return

    def on_completed(self) -> None:
        """Increment completion counter when the pipeline reports completion."""
        self.completed_count += 1

    def on_error(self, message: str) -> None:
        """Capture error messages reported by the pipeline.

        Args:
            message: Human-readable description of the error.
        """
        self.error_messages.append(message)


@pytest.fixture
def progress_recorder() -> _ProgressRecorder:
    """Provide a fresh progress recorder for each test.

    Returns:
        _ProgressRecorder: Instrumented progress callback instance.
    """
    return _ProgressRecorder()


def test_whisper_pipeline_merges_chunks_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch, progress_recorder: _ProgressRecorder
) -> None:
    """Pipeline should orchestrate chunking and merge chunk-level results."""
    chunk_responses = [
        {
            "text": "chunk one",
            "chunks": [
                {
                    "text": "first",
                    "timestamp": (0.0, 1.0),
                    "words": [
                        {"word": "first", "start": 0.1, "end": 0.9},
                    ],
                }
            ],
            "runtime_seconds": 1.5,
            "config_used": {"model": "stub", "chunk_length": 3},
        },
        {
            "text": "chunk two",
            "chunks": [
                {
                    "text": "second",
                    "timestamp": (0.0, 2.0),
                    "words": [
                        {"word": "second", "start": 0.2, "end": 1.8},
                    ],
                }
            ],
            "runtime_seconds": 2.0,
            "config_used": {"model": "stub", "chunk_length": 3},
        },
    ]

    backend = _RecordingBackend(responses=chunk_responses, chunk_length=3)

    cleanup_calls: list[str] = []

    def fake_ensure_wav(path: str) -> str:
        return "converted.wav"

    def fake_split_audio(
        audio_path: str,
        *,
        chunk_duration: float,
        chunk_overlap: float,
        min_chunk_duration: float = 1.0,
    ) -> list[tuple[str, float]]:
        assert audio_path == "converted.wav"
        assert chunk_duration == 3.0
        assert chunk_overlap == 0.0
        assert min_chunk_duration == 1.0
        return [("chunk_01.wav", 0.0), ("chunk_02.wav", 3.5)]

    def fake_cleanup(paths: list[str]) -> None:
        cleanup_calls.extend(paths)

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.conversion.ensure_wav", fake_ensure_wav
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.processing.split_audio", fake_split_audio
    )
    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.utils.file_utils.cleanup_temp_files", fake_cleanup
    )

    pipeline = WhisperPipeline(
        asr_backend=backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    result = pipeline.process(
        audio_file_path="input.mp3",
        language="en",
        task="transcribe",
        timestamp_type="chunk",
        progress_callback=progress_recorder,
    )

    assert result["text"] == "chunk one\n\nchunk two"
    # Timestamps are normalized to lists by merge_chunk_results
    assert result["chunks"][0]["timestamp"] == [0.0, 1.0]
    assert result["chunks"][1]["timestamp"] == [3.5, 5.5]
    assert result["chunks"][1]["words"][0]["start"] == pytest.approx(3.7)
    assert result["runtime_seconds"] == 3.5
    assert result["config_used"]["chunking_used"] is True
    assert result["config_used"]["num_chunks"] == 2

    assert backend.calls == [
        {
            "path": "chunk_01.wav",
            "language": "en",
            "task": "transcribe",
            "return_timestamps": True,
            "progress_cb_has_on_completed": True,
        },
        {
            "path": "chunk_02.wav",
            "language": "en",
            "task": "transcribe",
            "return_timestamps": True,
            "progress_cb_has_on_completed": True,
        },
    ]

    assert progress_recorder.audio_started == ["input.mp3"]
    assert progress_recorder.audio_finished == [None]
    assert progress_recorder.chunking_started == [2]
    assert progress_recorder.chunk_done == [0, 1]
    assert progress_recorder.inference_started == [2]
    assert progress_recorder.inference_batch_done == [0, 1]
    assert progress_recorder.completed_count == 1
    assert progress_recorder.error_messages == []

    assert sorted(cleanup_calls) == [
        "chunk_01.wav",
        "chunk_02.wav",
        "converted.wav",
    ]


def test_whisper_pipeline_word_timestamps_passthrough(
    monkeypatch: pytest.MonkeyPatch, progress_recorder: _ProgressRecorder
) -> None:
    """Pipeline should request word timestamps when configured."""
    backend = _RecordingBackend(
        responses=[
            {
                "text": "one chunk",
                "chunks": [],
                "runtime_seconds": 0.75,
                "config_used": {"model": "stub"},
            }
        ],
        chunk_length=15,
    )

    cleaned: list[str] = []

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.conversion.ensure_wav", lambda path: path
    )

    def single_chunk(
        path: str,
        *,
        chunk_duration: float,
        chunk_overlap: float,
        min_chunk_duration: float = 1.0,
    ) -> list[tuple[str, float]]:
        assert chunk_duration == 15.0
        assert chunk_overlap == 0.0
        assert min_chunk_duration == 1.0
        return [(path, 0.0)]

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.processing.split_audio",
        single_chunk,
    )

    def record_cleanup(paths: list[str]) -> None:
        cleaned.extend(paths)

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.utils.file_utils.cleanup_temp_files",
        record_cleanup,
    )

    pipeline = WhisperPipeline(
        asr_backend=backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    result = pipeline.process(
        audio_file_path="already.wav",
        language=None,
        task="translate",
        timestamp_type="word",
        progress_callback=progress_recorder,
    )

    assert result["text"] == "one chunk"
    assert result.get("chunks") == []
    assert backend.calls == [
        {
            "path": "already.wav",
            "language": None,
            "task": "translate",
            "return_timestamps": "word",
            "progress_cb_has_on_completed": True,
        }
    ]
    assert progress_recorder.completed_count == 1
    assert cleaned == []


def test_whisper_pipeline_accepts_bool_true_as_chunk_timestamps(
    monkeypatch: pytest.MonkeyPatch, progress_recorder: _ProgressRecorder
) -> None:
    """Pipeline should treat a legacy bool ``True`` timestamp flag as chunk-level."""
    backend = _RecordingBackend(
        responses=[
            {
                "text": "one chunk",
                "chunks": [],
                "runtime_seconds": 0.75,
                "config_used": {"model": "stub"},
            }
        ],
        chunk_length=15,
    )

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.conversion.ensure_wav", lambda path: path
    )

    def single_chunk(
        path: str,
        *,
        chunk_duration: float,
        chunk_overlap: float,
        min_chunk_duration: float = 1.0,
    ) -> list[tuple[str, float]]:
        assert chunk_duration == 15.0
        assert chunk_overlap == 0.0
        assert min_chunk_duration == 1.0
        return [(path, 0.0)]

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.audio.processing.split_audio",
        single_chunk,
    )

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.utils.file_utils.cleanup_temp_files",
        lambda _paths: None,
    )

    pipeline = WhisperPipeline(
        asr_backend=backend,
        storage_backend=None,
        save_transcriptions=False,
    )

    result = pipeline.process(
        audio_file_path="already.wav",
        language=None,
        task="translate",
        timestamp_type=True,
        progress_callback=progress_recorder,
    )

    assert result["text"] == "one chunk"
    assert backend.calls == [
        {
            "path": "already.wav",
            "language": None,
            "task": "translate",
            "return_timestamps": True,
            "progress_cb_has_on_completed": True,
        }
    ]
    assert progress_recorder.completed_count == 1
