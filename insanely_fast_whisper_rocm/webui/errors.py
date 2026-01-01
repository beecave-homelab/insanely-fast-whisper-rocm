"""Error classes for Insanely Fast Whisper API WebUI.

This module contains custom exception classes used throughout the WebUI
to provide specific error information and handling.
"""

from insanely_fast_whisper_rocm.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)

__all__ = [
    "DeviceNotFoundError",
    "ExportError",
    "FormatterError",
    "TranscriptionError",
]


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
