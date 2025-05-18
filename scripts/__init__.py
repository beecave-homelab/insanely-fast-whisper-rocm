"""
Command-line interface for the Insanely Fast Whisper ROCm project.

This package provides the command-line interface for the transcription service.
"""

# Import the main CLI function
from .cli import cli, transcribe, main

__all__ = ["cli", "transcribe", "main"]
