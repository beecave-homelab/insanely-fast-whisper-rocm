"""Custom exception classes for the ASR pipeline."""


class TranscriptionError(Exception):
    """Custom exception raised when transcription fails."""


class DeviceNotFoundError(Exception):
    """Custom exception raised when a requested compute device is not available."""
