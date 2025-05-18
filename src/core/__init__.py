"""
Core functionality for the Insanely Fast Whisper ROCm application.

This package contains the core modules for handling transcription, file operations,
and format conversions used throughout the application.
"""

from .transcription import (
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionProcessor,
    transcribe_file,
)

from .conversion import (
    OutputFormat,
    convert_transcription,
    batch_convert_directory,
    get_formatter,
)

from .file_handlers import (
    FileType,
    FileEventType,
    FileInfo,
    FileValidator,
    DirectoryWatcher,
    ensure_directory,
    safe_move,
    find_files,
)

__all__ = [
    # Transcription
    "TranscriptionConfig",
    "TranscriptionResult",
    "TranscriptionProcessor",
    "transcribe_file",
    # Conversion
    "OutputFormat",
    "convert_transcription",
    "batch_convert_directory",
    "get_formatter",
    # File Handlers
    "FileType",
    "FileEventType",
    "FileInfo",
    "FileValidator",
    "DirectoryWatcher",
    "ensure_directory",
    "safe_move",
    "find_files",
]
