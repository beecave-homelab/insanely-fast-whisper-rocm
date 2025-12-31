"""Error classes for Insanely Fast Whisper API WebUI.

This module contains custom exception classes used throughout the WebUI
to provide specific error information and handling.
"""


class TranscriptionError(Exception):  # pylint: disable=too-few-public-methods
    """Custom exception raised when transcription fails in the WebUI context.

    This exception provides more specific error information than the base
    ClickException and is used throughout the ASR pipeline to indicate
    transcription-specific failures.
    """


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


class DeviceNotFoundError(Exception):  # pylint: disable=too-few-public-methods
    """Custom exception raised when a requested compute device is unavailable.

    This exception is raised when the application attempts to use a specific
    device (e.g., CUDA GPU, MPS) that is either not present or not properly
    configured on the system in the WebUI context.
    """


class FormatterError(Exception):
    """Custom exception raised when formatting transcription results fails.

    This exception is used when there is an error in formatting the
    transcription results into a specific output format.
    """


class ExportError(Exception):
    """Custom exception raised when exporting transcription results fails.

    This exception is used when there is an error in exporting the
    transcription results to a file or other output format.
    """
