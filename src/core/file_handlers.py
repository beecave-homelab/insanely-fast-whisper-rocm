"""
Core module for handling file system operations and directory monitoring.

This module provides functionality for file validation, directory monitoring,
and other file system operations needed for the transcription service.
"""

import os
import time
import hashlib
import logging
import shutil
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import fnmatch
import json

from pydantic import BaseModel, Field, validator, DirectoryPath, FilePath
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileSystemEvent,
    FileSystemMovedEvent,
)

# Configure logging
logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types for the transcription service."""

    AUDIO = auto()
    VIDEO = auto()
    SUBTITLE = auto()
    TRANSCRIPT = auto()
    UNKNOWN = auto()


class FileEventType(Enum):
    """Types of file system events that can be monitored."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileInfo:
    """Metadata and information about a file."""

    path: Path
    size: int
    last_modified: float
    file_type: FileType
    mime_type: str
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FileValidator:
    """Validates files based on type, size, and other criteria."""

    # Common audio/video MIME types
    AUDIO_MIME_TYPES = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/x-m4a",
        "audio/mp4",
        "audio/x-aiff",
        "audio/flac",
        "audio/webm",
    }

    VIDEO_MIME_TYPES = {
        "video/mp4",
        "video/x-matroska",
        "video/webm",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-ms-wmv",
        "video/mpeg",
    }

    SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
    TRANSCRIPT_EXTENSIONS = {".json", ".txt"}

    def __init__(
        self,
        max_file_size: int = 1024 * 1024 * 500,  # 500MB default
        allowed_extensions: Optional[Set[str]] = None,
        allowed_mime_types: Optional[Set[str]] = None,
    ):
        """Initialize the file validator.

        Args:
            max_file_size: Maximum allowed file size in bytes.
            allowed_extensions: Set of allowed file extensions (with leading dot).
            allowed_mime_types: Set of allowed MIME types.
        """
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or set()
        self.allowed_mime_types = allowed_mime_types or set()

        # Initialize MIME types
        mimetypes.init()

    def get_file_type(self, path: Union[str, Path]) -> FileType:
        """Determine the type of a file based on its extension and MIME type."""
        path = Path(path)
        ext = path.suffix.lower()

        # First check by extension
        if ext in self.SUBTITLE_EXTENSIONS:
            return FileType.SUBTITLE

        if ext in self.TRANSCRIPT_EXTENSIONS:
            return FileType.TRANSCRIPT

        # Then check MIME type
        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type:
            if mime_type.startswith("audio/"):
                return FileType.AUDIO
            elif mime_type.startswith("video/"):
                return FileType.VIDEO

        # Fallback to extension for common cases
        audio_extensions = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac", ".wma"}
        video_extensions = {
            ".mp4",
            ".mkv",
            ".webm",
            ".mov",
            ".avi",
            ".wmv",
            ".mpeg",
            ".mpg",
        }

        if ext in audio_extensions:
            return FileType.AUDIO
        elif ext in video_extensions:
            return FileType.VIDEO

        return FileType.UNKNOWN

    def is_valid_media_file(self, path: Union[str, Path]) -> bool:
        """Check if a file is a valid audio or video file."""
        file_type = self.get_file_type(path)
        return file_type in (FileType.AUDIO, FileType.VIDEO)

    def is_valid_subtitle_file(self, path: Union[str, Path]) -> bool:
        """Check if a file is a valid subtitle file."""
        return self.get_file_type(path) == FileType.SUBTITLE

    def is_valid_transcript_file(self, path: Union[str, Path]) -> bool:
        """Check if a file is a valid transcript file."""
        return self.get_file_type(path) == FileType.TRANSCRIPT

    def validate_file(
        self,
        path: Union[str, Path],
        check_exists: bool = True,
        check_size: bool = True,
        check_type: bool = True,
        expected_type: Optional[FileType] = None,
    ) -> Tuple[bool, str]:
        """Validate a file against various criteria.

        Args:
            path: Path to the file to validate.
            check_exists: Whether to check if the file exists.
            check_size: Whether to check the file size.
            check_type: Whether to check the file type.
            expected_type: If provided, check if the file matches this type.

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(path)

        # Check if file exists
        if check_exists and not path.exists():
            return False, f"File does not exist: {path}"

        # Check if it's a file (not a directory)
        if check_exists and not path.is_file():
            return False, f"Path is not a file: {path}"

        # Check file size
        if check_size and check_exists:
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return (
                    False,
                    f"File size {file_size} exceeds maximum allowed {self.max_file_size}",
                )

        # Check file type
        if check_type and check_exists:
            # Check allowed extensions if specified
            if (
                self.allowed_extensions
                and path.suffix.lower() not in self.allowed_extensions
            ):
                return (
                    False,
                    f"File extension {path.suffix} not in allowed extensions: {self.allowed_extensions}",
                )

            # Check MIME type if specified
            if self.allowed_mime_types:
                mime_type, _ = mimetypes.guess_type(str(path))
                if mime_type not in self.allowed_mime_types:
                    return (
                        False,
                        f"MIME type {mime_type} not in allowed MIME types: {self.allowed_mime_types}",
                    )

            # Check expected file type if specified
            if expected_type is not None:
                actual_type = self.get_file_type(path)
                if actual_type != expected_type:
                    return (
                        False,
                        f"Expected file type {expected_type.name}, got {actual_type.name}",
                    )

        return True, ""

    def get_file_info(self, path: Union[str, Path]) -> Optional[FileInfo]:
        """Get detailed information about a file."""
        try:
            path = Path(path)
            stat = path.stat()

            file_type = self.get_file_type(path)
            mime_type, _ = mimetypes.guess_type(str(path))

            return FileInfo(
                path=path,
                size=stat.st_size,
                last_modified=stat.st_mtime,
                file_type=file_type,
                mime_type=mime_type or "application/octet-stream",
            )
        except Exception as e:
            logger.error(f"Error getting file info for {path}: {str(e)}")
            return None

    def calculate_checksum(
        self, path: Union[str, Path], algorithm: str = "md5"
    ) -> Optional[str]:
        """Calculate the checksum of a file."""
        path = Path(path)
        if not path.exists() or not path.is_file():
            return None

        hash_func = getattr(hashlib, algorithm, None)
        if hash_func is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        h = hash_func()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            return h.hexdigest()
        except IOError as e:
            logger.error(f"Error calculating checksum for {path}: {str(e)}")
            return None


class DirectoryWatcher:
    """Monitors a directory for file system events."""

    def __init__(
        self,
        watch_dir: Union[str, Path],
        recursive: bool = True,
        event_handlers: Optional[
            Dict[FileEventType, List[Callable[[FileSystemEvent], None]]]
        ] = None,
        file_validator: Optional[FileValidator] = None,
    ):
        """Initialize the directory watcher.

        Args:
            watch_dir: Directory to watch for changes.
            recursive: Whether to watch subdirectories recursively.
            event_handlers: Dictionary mapping event types to lists of handler functions.
            file_validator: Optional FileValidator instance to validate files.
        """
        self.watch_dir = Path(watch_dir).resolve()
        self.recursive = recursive
        self.observer = Observer()
        self.event_handlers = event_handlers or {}
        self.file_validator = file_validator or FileValidator()
        self.running = False

        # Ensure watch directory exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)

    def add_event_handler(
        self,
        event_type: FileEventType,
        handler: Callable[[FileSystemEvent], None],
    ) -> None:
        """Add an event handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def remove_event_handler(
        self,
        event_type: FileEventType,
        handler: Callable[[FileSystemEvent], None],
    ) -> None:
        """Remove an event handler."""
        if event_type in self.event_handlers:
            if handler in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(handler)

    def _on_any_event(self, event: FileSystemEvent) -> None:
        """Internal method called on any file system event."""
        try:
            # Skip directories and invalid files
            if event.is_directory:
                return

            # Determine event type
            event_type = None
            if event.event_type == "created":
                event_type = FileEventType.CREATED
            elif event.event_type == "modified":
                event_type = FileEventType.MODIFIED
            elif event.event_type == "deleted":
                event_type = FileEventType.DELETED
            elif event.event_type == "moved":
                event_type = FileEventType.MOVED

            if not event_type:
                return

            # Call all registered handlers for this event type
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(
                            f"Error in event handler for {event_type}: {str(e)}"
                        )
                        logger.debug(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error processing file system event: {str(e)}")
            logger.debug(traceback.format_exc())

    def start(self) -> None:
        """Start watching the directory."""
        if self.running:
            logger.warning("Directory watcher is already running")
            return

        # Create event handler
        event_handler = FileSystemEventHandler()
        event_handler.on_any_event = self._on_any_event

        # Schedule the observer
        self.observer.schedule(
            event_handler, str(self.watch_dir), recursive=self.recursive
        )

        # Start the observer
        self.observer.start()
        self.running = True
        logger.info(f"Started watching directory: {self.watch_dir}")

    def stop(self) -> None:
        """Stop watching the directory."""
        if not self.running:
            return

        self.observer.stop()
        self.observer.join()
        self.running = False
        logger.info(f"Stopped watching directory: {self.watch_dir}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.
        mode: Permissions to set on the directory (octal).

    Returns:
        Path object for the directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    path = Path(path)
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    return path


def safe_move(
    src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False
) -> bool:
    """Safely move a file, creating parent directories if needed.

    Args:
        src: Source file path.
        dst: Destination file path.
        overwrite: Whether to overwrite the destination if it exists.

    Returns:
        True if the move was successful, False otherwise.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        logger.error(f"Source file does not exist: {src}")
        return False

    if dst.exists():
        if not overwrite:
            logger.error(f"Destination file already exists and overwrite=False: {dst}")
            return False
        try:
            dst.unlink()
        except OSError as e:
            logger.error(f"Failed to remove existing destination file {dst}: {str(e)}")
            return False

    # Ensure destination directory exists
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create destination directory {dst.parent}: {str(e)}")
        return False

    # Perform the move
    try:
        shutil.move(str(src), str(dst))
        return True
    except (shutil.Error, OSError) as e:
        logger.error(f"Failed to move {src} to {dst}: {str(e)}")
        return False


def find_files(
    directory: Union[str, Path],
    patterns: Optional[Union[str, List[str]]] = None,
    recursive: bool = True,
    file_type: Optional[FileType] = None,
    validator: Optional[FileValidator] = None,
) -> List[Path]:
    """Find files matching the given patterns and type.

    Args:
        directory: Directory to search in.
        patterns: Glob patterns to match (e.g., '*.wav' or ['*.mp3', '*.wav']).
        recursive: Whether to search recursively in subdirectories.
        file_type: If specified, only return files of this type.
        validator: Optional FileValidator to use for type checking.

    Returns:
        List of matching file paths.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []

    if isinstance(patterns, str):
        patterns = [patterns]

    validator = validator or FileValidator()
    results = set()

    # If no patterns specified, use all files
    if not patterns:
        patterns = ["**/*"] if recursive else ["*"]

    # Find files matching patterns
    for pattern in patterns:
        # Handle recursive patterns
        if recursive and "**" not in pattern:
            pattern = f"**/{pattern}"

        # Find all matches for this pattern
        for match in directory.glob(pattern):
            if not match.is_file():
                continue

            # Filter by file type if specified
            if file_type is not None:
                actual_type = validator.get_file_type(match)
                if actual_type != file_type:
                    continue

            results.add(match.resolve())

    return sorted(results)
