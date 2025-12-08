"""Progress reporting protocol for CLI and backend.

Defines a typed callback interface to report progress across major phases of the
CLI workflow and backend processing. A no-op implementation is provided for
callers that do not require progress reporting.
"""

from __future__ import annotations

from typing import Protocol


class ProgressCallback(Protocol):
    """Typed protocol for progress reporting.

    Each method should be safe to call multiple times and may be a no-op when
    the implementing reporter decides to ignore certain phases.
    """

    # Model loading
    def on_model_load_started(self) -> None:
        """Signal that model loading has started."""

    def on_model_load_finished(self) -> None:
        """Signal that model loading has finished."""

    # Audio I/O
    def on_audio_loading_started(self, path: str) -> None:
        """Signal that audio loading/conversion has started.

        Args:
            path: Filesystem path to the input audio file.
        """

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:
        """Signal that audio loading/conversion has finished.

        Args:
            duration_sec: Optional duration in seconds if known; otherwise None.
        """

    # Chunking/inference
    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Signal that chunking has started.

        Args:
            total_chunks: Total number of chunks if known; otherwise None.
        """

    def on_chunk_done(self, index: int) -> None:
        """Signal that a single chunk has been processed.

        Args:
            index: Zero-based index of the completed chunk.
        """

    def on_inference_started(self, total_batches: int | None) -> None:
        """Signal that inference has started.

        Args:
            total_batches: Total number of batches if known; otherwise None.
        """

    def on_inference_batch_done(self, index: int) -> None:
        """Signal that a single inference batch has completed.

        Args:
            index: Zero-based index of the completed batch.
        """

    # Post-processing
    def on_postprocess_started(self, name: str) -> None:
        """Signal start of a named post-processing step.

        Args:
            name: Short identifier for the step (e.g., "stable-ts").
        """

    def on_postprocess_finished(self, name: str) -> None:
        """Signal end of a named post-processing step.

        Args:
            name: Short identifier for the step (e.g., "stable-ts").
        """

    # Export
    def on_export_started(self, total_items: int) -> None:
        """Signal that export has started with a known total item count.

        Args:
            total_items: Total number of items (files) to export.
        """

    def on_export_item_done(self, index: int, label: str) -> None:
        """Signal that a single export item has been completed.

        Args:
            index: Zero-based index of the completed item.
            label: Short label for the item (e.g., "json", "srt").
        """

    # Terminal states
    def on_completed(self) -> None:
        """Signal that the entire operation has completed."""

    def on_error(self, message: str) -> None:
        """Signal that an unrecoverable error has occurred.

        Args:
            message: Human-readable error message.
        """


class NoOpProgress(ProgressCallback):
    """No-op implementation of ProgressCallback."""

    def on_model_load_started(self) -> None:
        """Do nothing when model load starts."""
        pass

    def on_model_load_finished(self) -> None:
        """Do nothing when model load finishes."""
        pass

    def on_audio_loading_started(self, path: str) -> None:
        """Do nothing when audio loading starts."""
        pass

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:
        """Do nothing when audio loading finishes."""
        pass

    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Do nothing when chunking starts."""
        pass

    def on_chunk_done(self, index: int) -> None:
        """Do nothing when a chunk is processed."""
        pass

    def on_inference_started(self, total_batches: int | None) -> None:
        """Do nothing when inference starts."""
        pass

    def on_inference_batch_done(self, index: int) -> None:
        """Do nothing when an inference batch completes."""
        pass

    def on_postprocess_started(self, name: str) -> None:
        """Do nothing when a post-processing step starts."""
        pass

    def on_postprocess_finished(self, name: str) -> None:
        """Do nothing when a post-processing step finishes."""
        pass

    def on_export_started(self, total_items: int) -> None:
        """Do nothing when export starts."""
        pass

    def on_export_item_done(self, index: int, label: str) -> None:
        """Do nothing when an export item completes."""
        pass

    def on_completed(self) -> None:
        """Do nothing when the operation completes."""
        pass

    def on_error(self, message: str) -> None:
        """Do nothing when an unrecoverable error occurs."""
        pass
