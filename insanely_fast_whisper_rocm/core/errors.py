"""Custom exception classes for the ASR pipeline."""

from __future__ import annotations


class TranscriptionError(Exception):
    """Custom exception raised when transcription fails."""


class OutOfMemoryError(TranscriptionError):
    """Base class for Out of Memory errors."""

    def __init__(
        self,
        message: str,
        device: str | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize the OutOfMemoryError.

        Args:
            message: Error message.
            device: Optional device identifier.
            config: Optional configuration dictionary.
        """
        super().__init__(message)
        self.device = device
        self.config = config


class ModelLoadingOOMError(OutOfMemoryError):
    """Raised when model initialization fails due to OOM."""


class InferenceOOMError(OutOfMemoryError):
    """Raised when audio processing fails due to OOM."""


class TranscriptionCancelledError(TranscriptionError):
    """Raised when transcription is cancelled by the caller."""


class DiarizationError(TranscriptionError):
    """Raised when speaker diarization fails.

    Attributes:
        model: The diarization model name that was requested.
        reason: Short description of why diarization failed.
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Initialize the DiarizationError.

        Args:
            message: Error message.
            model: Optional diarization model identifier.
            reason: Optional short reason for the failure.
        """
        super().__init__(message)
        self.model = model
        self.reason = reason


class DeviceNotFoundError(TranscriptionError):
    """Custom exception raised when a requested compute device is not available."""
