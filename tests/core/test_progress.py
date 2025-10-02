"""Tests for progress reporting protocol and implementations.

Validates that NoOpProgress correctly implements the ProgressCallback protocol
and that all methods execute without raising exceptions.
"""

from __future__ import annotations

from insanely_fast_whisper_api.core.progress import NoOpProgress, ProgressCallback


def test_noop_progress__implements_protocol() -> None:
    """NoOpProgress should satisfy the ProgressCallback protocol."""
    progress: ProgressCallback = NoOpProgress()
    assert isinstance(progress, NoOpProgress)


def test_noop_progress__on_model_load_started__does_not_raise() -> None:
    """NoOpProgress.on_model_load_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_model_load_started()


def test_noop_progress__on_model_load_finished__does_not_raise() -> None:
    """NoOpProgress.on_model_load_finished should execute without raising."""
    progress = NoOpProgress()
    progress.on_model_load_finished()


def test_noop_progress__on_audio_loading_started__does_not_raise() -> None:
    """NoOpProgress.on_audio_loading_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_audio_loading_started("/path/to/audio.wav")


def test_noop_progress__on_audio_loading_finished__does_not_raise() -> None:
    """NoOpProgress.on_audio_loading_finished should execute without raising."""
    progress = NoOpProgress()
    progress.on_audio_loading_finished(None)
    progress.on_audio_loading_finished(120.5)


def test_noop_progress__on_chunking_started__does_not_raise() -> None:
    """NoOpProgress.on_chunking_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_chunking_started(None)
    progress.on_chunking_started(10)


def test_noop_progress__on_chunk_done__does_not_raise() -> None:
    """NoOpProgress.on_chunk_done should execute without raising."""
    progress = NoOpProgress()
    progress.on_chunk_done(0)
    progress.on_chunk_done(5)


def test_noop_progress__on_inference_started__does_not_raise() -> None:
    """NoOpProgress.on_inference_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_inference_started(None)
    progress.on_inference_started(8)


def test_noop_progress__on_inference_batch_done__does_not_raise() -> None:
    """NoOpProgress.on_inference_batch_done should execute without raising."""
    progress = NoOpProgress()
    progress.on_inference_batch_done(0)
    progress.on_inference_batch_done(3)


def test_noop_progress__on_postprocess_started__does_not_raise() -> None:
    """NoOpProgress.on_postprocess_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_postprocess_started("stable-ts")


def test_noop_progress__on_postprocess_finished__does_not_raise() -> None:
    """NoOpProgress.on_postprocess_finished should execute without raising."""
    progress = NoOpProgress()
    progress.on_postprocess_finished("stable-ts")


def test_noop_progress__on_export_started__does_not_raise() -> None:
    """NoOpProgress.on_export_started should execute without raising."""
    progress = NoOpProgress()
    progress.on_export_started(3)


def test_noop_progress__on_export_item_done__does_not_raise() -> None:
    """NoOpProgress.on_export_item_done should execute without raising."""
    progress = NoOpProgress()
    progress.on_export_item_done(0, "json")
    progress.on_export_item_done(1, "srt")


def test_noop_progress__on_completed__does_not_raise() -> None:
    """NoOpProgress.on_completed should execute without raising."""
    progress = NoOpProgress()
    progress.on_completed()


def test_noop_progress__on_error__does_not_raise() -> None:
    """NoOpProgress.on_error should execute without raising."""
    progress = NoOpProgress()
    progress.on_error("Test error message")


def test_noop_progress__all_methods__can_be_called_in_sequence() -> None:
    """All NoOpProgress methods should work in a typical usage sequence."""
    progress = NoOpProgress()

    # Model loading phase
    progress.on_model_load_started()
    progress.on_model_load_finished()

    # Audio loading phase
    progress.on_audio_loading_started("/path/to/audio.mp3")
    progress.on_audio_loading_finished(180.0)

    # Chunking phase
    progress.on_chunking_started(5)
    for i in range(5):
        progress.on_chunk_done(i)

    # Inference phase
    progress.on_inference_started(10)
    for i in range(10):
        progress.on_inference_batch_done(i)

    # Post-processing phase
    progress.on_postprocess_started("stable-ts")
    progress.on_postprocess_finished("stable-ts")

    # Export phase
    progress.on_export_started(3)
    progress.on_export_item_done(0, "json")
    progress.on_export_item_done(1, "srt")
    progress.on_export_item_done(2, "txt")

    # Completion
    progress.on_completed()
