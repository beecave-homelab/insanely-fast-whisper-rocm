"""Rich-based progress reporter implementation for the CLI.

This module provides a concrete implementation of the ProgressCallback protocol
using Rich's Progress UI. Rendering is automatically disabled when the terminal
is not a TTY or when the reporter is initialized with enabled=False.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from insanely_fast_whisper_api.core.progress import ProgressCallback


@dataclass
class _TaskHandles:
    model: TaskID | None = None
    audio: TaskID | None = None
    transcription: TaskID | None = None  # unified percentage bar
    postprocess: TaskID | None = None
    export: TaskID | None = None


class RichProgressReporter(ProgressCallback):
    """Render progress using Rich, with phase-specific tasks.

    Args:
        enabled: Whether to render progress. If False or if stdout is not a TTY,
            no progress will be displayed and all methods become no-ops.
    """

    def __init__(self, *, enabled: bool = True, show_messages: bool = True) -> None:
        """Initialize the reporter.

        Args:
            enabled: Whether to render progress when stdout is a TTY.
            show_messages: Whether to display auxiliary one-line messages like
                "Model loaded" and "Audio loaded". Set to False when the CLI
                is in quiet mode to keep output minimal.
        """
        self.console = Console()
        self.enabled = enabled and sys.stdout.isatty()
        self._show_messages = show_messages
        self._progress: Progress | None = None
        self._tasks = _TaskHandles()
        self._total_chunks: int | None = None

        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                expand=True,
                transient=True,
            )
            self._progress.start()

    # ----------------------- lifecycle helpers ----------------------- #
    def _stop(self) -> None:
        """Stop the progress display if active."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Destructor for cleanup."""
        try:
            self._stop()
        except Exception:  # noqa: BLE001
            pass

    # --------------------------- model load -------------------------- #
    def on_model_load_started(self) -> None:
        """No-op (avoid spinner); we print a checkmark at finish instead."""
        return

    def on_model_load_finished(self) -> None:
        """Print a one-line model loaded checkmark."""
        if not self.enabled:
            return
        self.console.print("[green]✔ Model loaded[/]")

    # ---------------------------- audio I/O --------------------------- #
    def on_audio_loading_started(self, path: str) -> None:
        """No-op (avoid spinner); we print a checkmark at finish instead."""
        return

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:
        """Print a one-line audio loaded checkmark."""
        if not self.enabled:
            return
        self.console.print("[green]✔ Audio prepared.[/]")

    # ---------------------- unified transcription --------------------- #
    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Create the unified transcription progress bar.

        Args:
            total_chunks: Total number of chunks to process.
        """
        if not self.enabled or self._progress is None:
            return
        self._total_chunks = int(total_chunks or 0)
        if self._tasks.transcription is None:
            self._tasks.transcription = self._progress.add_task(
                "Transcription", total=self._total_chunks
            )

    def on_chunk_done(self, index: int) -> None:
        """Advance transcription progress by one chunk.

        Args:
            index: Chunk index.
        """
        if not self.enabled or self._progress is None:
            return
        # Advance the bar by one chunk
        if self._tasks.transcription is not None:
            self._progress.advance(self._tasks.transcription, 1)
        # Stop the task when it completes to avoid duplicate lingering bars
        task_id = self._tasks.transcription
        if task_id is not None:
            task = self._progress.tasks[self._progress.task_ids.index(task_id)]
            if task.total is not None and task.completed >= task.total:
                # Ensure bar shows 100% and is cleared before printing the checkmark
                try:
                    # Snap to full just in case of rounding
                    self._progress.update(task_id, completed=task.total)
                    self._progress.refresh()
                except Exception:  # noqa: BLE001
                    pass
                # Stop/hide the task prior to emitting the standalone line
                self._progress.stop_task(task_id)
                self._tasks.transcription = None
                self.console.print("[green]✔ Transcription[/]")

    # --------------------------- inference ---------------------------- #
    def on_inference_started(self, total_batches: int | None) -> None:
        """Start inference progress (no-op for unified bar)."""
        # No-op: percentage bar is driven by chunks
        return

    def on_inference_batch_done(self, index: int) -> None:
        """Finish inference batch (no-op for unified bar).

        Args:
            index: Batch index.
        """
        # No-op: percentage bar is driven by chunks
        return

    # ------------------------ post-processing ------------------------- #
    def on_postprocess_started(self, name: str) -> None:
        """Start post-processing progress.

        Args:
            name: Name of the post-processing step.
        """
        if not self.enabled or self._progress is None:
            return
        if self._tasks.postprocess is None:
            self._tasks.postprocess = self._progress.add_task(
                f"Post: {name}", total=None
            )

    def on_postprocess_finished(self, name: str) -> None:  # noqa: ARG002
        """Finish post-processing progress.

        Args:
            name: Name of the post-processing step.
        """
        if (
            not self.enabled
            or self._progress is None
            or self._tasks.postprocess is None
        ):
            return
        self._progress.update(self._tasks.postprocess, completed=1)
        self._progress.stop_task(self._tasks.postprocess)

    # ------------------------------ export ---------------------------- #
    def on_export_started(self, total_items: int) -> None:
        """Start export progress.

        Args:
            total_items: Total number of items to export.
        """
        if not self.enabled or self._progress is None:
            return
        if self._tasks.export is None:
            self._tasks.export = self._progress.add_task("Export", total=total_items)

    def on_export_item_done(self, index: int, label: str) -> None:  # noqa: ARG002
        """Advance export progress.

        Args:
            index: Item index.
            label: Item label.
        """
        if not self.enabled or self._progress is None or self._tasks.export is None:
            return
        self._progress.advance(self._tasks.export, 1)

    # ---------------------------- terminal ---------------------------- #
    def on_completed(self) -> None:
        """Stop all progress and clean up."""
        self._stop()

    def on_error(self, message: str) -> None:  # noqa: ARG002
        """Handle error by stopping progress.

        Args:
            message: Error message.
        """
        self._stop()
