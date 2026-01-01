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
        """
        Initialize the OutOfMemoryError with an error message and optional runtime context.
        
        Parameters:
            message (str): Human-readable error message describing the out-of-memory condition.
            device (str | None): Optional device identifier where the error occurred (e.g., "cpu", "cuda:0").
            config (dict | None): Optional configuration or runtime settings present when the error occurred.
        
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


class DeviceNotFoundError(TranscriptionError):
    """Custom exception raised when a requested compute device is not available."""