"""tqdm-based progress reporter implementation for the CLI.

This module provides a concrete implementation of the ProgressCallback protocol
using tqdm progress bars and messages printed via ``tqdm.write``.
It mirrors the UX previously provided by the Rich-based reporter:

- "✔ Model loaded"
- "✔ Audio prepared."
- A unified "Transcription" bar that reaches 100% before printing
  "✔ Transcription"
- Optional post-processing and export progress
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm

from insanely_fast_whisper_api.core.progress import ProgressCallback


@dataclass
class _Bars:
    """Holds active tqdm bars.

    Attributes:
        transcription: Bar for the unified transcription phase.
        postprocess: Reserved for post-processing (unused; we print lines).
        export: Bar for exporting multiple items.
    """

    transcription: tqdm | None = None
    postprocess: tqdm | None = None
    export: tqdm | None = None


class TqdmProgressReporter(ProgressCallback):
    """Render progress using tqdm, with phase-specific bars and messages.

    Args:
        enabled: Whether to render progress. If False, all methods become no-ops.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        """Initialize the tqdm reporter.

        Args:
            enabled: Enable or disable progress output.
        """
        self.enabled = enabled
        self._bars = _Bars()
        self._total_chunks: int | None = None
        self._export_single: bool = False

    # --------------------------- model load -------------------------- #
    def on_model_load_started(self) -> None:
        """No-op before model load in tqdm mode."""
        return

    def on_model_load_finished(self) -> None:
        """Print a one-line model loaded checkmark."""
        if not self.enabled:
            return
        tqdm.write("✔ Model loaded")

    # ---------------------------- audio I/O --------------------------- #
    def on_audio_loading_started(self, path: str) -> None:  # noqa: ARG002
        """No-op before audio load in tqdm mode."""
        return

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:  # noqa: ARG002
        """Print a one-line audio prepared checkmark."""
        if not self.enabled:
            return
        tqdm.write("✔ Audio prepared.")

    # ---------------------- unified transcription --------------------- #
    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Create the unified transcription tqdm bar."""
        if not self.enabled:
            return
        self._total_chunks = int(total_chunks or 0)
        if self._total_chunks <= 0:
            return
        if self._bars.transcription is None:
            self._bars.transcription = tqdm(
                total=self._total_chunks,
                desc="Transcription",
                leave=False,
            )

    def on_chunk_done(self, index: int) -> None:  # noqa: ARG002
        """Advance the transcription bar and print completion when done."""
        if not self.enabled:
            return
        bar = self._bars.transcription
        if bar is not None:
            bar.update(1)
            # When finished, snap to full, close, then print the checkmark line
            if bar.total is not None and bar.n >= bar.total:
                try:
                    bar.n = bar.total  # ensure visually 100%
                    bar.refresh()
                finally:
                    bar.close()
                    self._bars.transcription = None
                    tqdm.write("✔ Transcription")

    # --------------------------- inference ---------------------------- #
    def on_inference_started(self, total_batches: int | None) -> None:  # noqa: ARG002
        """No-op: unified bar is driven by chunks."""
        return

    def on_inference_batch_done(self, index: int) -> None:  # noqa: ARG002
        """No-op: unified bar is driven by chunks."""
        return

    # ------------------------ post-processing ------------------------- #
    def on_postprocess_started(self, name: str) -> None:
        """No-op to keep post-process output minimal and clean."""
        if not self.enabled:
            return
        return

    def on_postprocess_finished(self, name: str) -> None:
        """Print granular completion lines for Demucs and VAD.

        Args:
            name: Identifier for the post-process step. Supported values:
                - "demucs" → prints "✔ Demucs vocals isolated".
                - "vad threshold=..." → prints threshold value.
                - any other → generic "✔ Post completed".
        """
        if not self.enabled:
            return
        key = name.strip().lower()
        if key == "demucs":
            tqdm.write("✔ Demucs vocals isolated")
            return
        if key.startswith("vad"):
            # Expected formats: "vad" or "vad threshold=0.35"
            threshold = None
            if "threshold=" in key:
                try:
                    threshold = key.split("threshold=", 1)[1]
                except Exception:  # noqa: BLE001
                    threshold = None
            if threshold:
                tqdm.write(f"✔ VAD applied (threshold={threshold})")
            else:
                tqdm.write("✔ VAD applied")
            return
        tqdm.write("✔ Post completed")

    # ------------------------------ export ---------------------------- #
    def on_export_started(self, total_items: int) -> None:
        """Start export progress, suppress bar if only one item.

        For a single export, we prefer a concise checkmark line with the
        destination path instead of a tqdm bar.
        """
        if not self.enabled:
            return
        self._export_single = total_items <= 1
        if not self._export_single and self._bars.export is None:
            self._bars.export = tqdm(total=total_items, desc="Export", leave=False)

    def on_export_item_done(self, index: int, label: str) -> None:  # noqa: ARG002
        """Advance export bar or print a single-item checkmark line.

        If ``on_export_started`` determined a single export item, ``label`` can
        embed the destination path using the convention "FMT::/full/path" to
        allow a concise line "✔ Export FMT to: /full/path".
        """
        if not self.enabled:
            return
        if self._export_single:
            # Suppress extra line for single export to avoid duplicate messages;
            # the CLI already prints a "Saved ..." line.
            return
        bar = self._bars.export
        if bar is not None:
            bar.update(1)
            if bar.total is not None and bar.n >= bar.total:
                try:
                    bar.n = bar.total
                    bar.refresh()
                finally:
                    bar.close()
                    self._bars.export = None

    # ---------------------------- terminal ---------------------------- #
    def on_completed(self) -> None:
        """Close any open bars defensively."""
        for attr in ("transcription", "postprocess", "export"):
            bar = getattr(self._bars, attr)
            if bar is not None:
                try:
                    bar.close()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    setattr(self._bars, attr, None)

    def on_error(self, message: str) -> None:  # noqa: ARG002
        """Close bars on error too."""
        self.on_completed()
