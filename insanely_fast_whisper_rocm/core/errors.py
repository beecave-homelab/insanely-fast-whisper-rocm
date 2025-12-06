"""Custom exception classes for the ASR pipeline."""

from __future__ import annotations


class TranscriptionError(Exception):
    """Custom exception raised when transcription fails."""


class TranscriptionCancelledError(TranscriptionError):
    """Raised when transcription is cancelled by the caller."""


class DeviceNotFoundError(Exception):
    """Custom exception raised when a requested compute device is not available."""
