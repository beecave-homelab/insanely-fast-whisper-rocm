"""Error classes for Insanely Fast Whisper API WebUI.

This module contains custom exception classes used throughout the WebUI
to provide specific error information and handling.
"""

# import click # Removing click dependency from webui errors


class TranscriptionError(Exception):  # pylint: disable=too-few-public-methods
    """Custom exception raised when transcription fails in the WebUI context.

    This exception provides more specific error information than the base ClickException
    and is used throughout the ASR pipeline to indicate transcription-specific failures.
    """


class DeviceNotFoundError(Exception):  # pylint: disable=too-few-public-methods
    """Custom exception raised when a requested compute device is not available in the WebUI context.

    This exception is raised when the application attempts to use a specific
    device (e.g., CUDA GPU, MPS) that is either not present or not properly
    configured on the system.
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
