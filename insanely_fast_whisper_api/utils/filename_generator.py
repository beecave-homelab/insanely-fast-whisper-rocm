"""Utilities for generating standardized filenames for transcription outputs."""

import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from insanely_fast_whisper_api.utils.constants import APP_TIMEZONE


class TaskType(Enum):
    """Enumeration for task types."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


@dataclass
class FilenameComponents:
    """Data class to hold components for filename generation."""

    audio_stem: str
    task: TaskType
    timestamp: datetime.datetime
    extension: str
    # interface_context: str | None = None # Placeholder for future interface-specific needs


class FilenameGenerationStrategy(ABC):
    """Abstract base class for filename generation strategies."""

    @abstractmethod
    def generate_filename(self, components: FilenameComponents) -> str:
        """Generates a filename based on the provided components.

        Args:
            components: An instance of FilenameComponents containing all
                        necessary parts to construct the filename.

        Returns:
            A string representing the generated filename.
        """
        raise NotImplementedError


class StandardFilenameStrategy(FilenameGenerationStrategy):
    """Standard strategy for generating filenames.
    Pattern: {audio_stem}_{task}_{timestamp}.{extension}
    Timestamp format: YYYYMMDDTHHMMSSZ (ISO 8601 like)
    """

    def generate_filename(self, components: FilenameComponents) -> str:
        """Generates a filename using the standard unified pattern.
        Example: my_audio_transcribe_20241201T143022Z.json
        """
        # Ensure UTC timestamp if not already, matching 'Z' suffix.
        # The calling code in FilenameGenerator ensures UTC for .now()
        timestamp_str = components.timestamp.strftime("%Y%m%dT%H%M%SZ")

        # Basic sanitization for extension (remove leading dot if present)
        clean_extension = components.extension.lstrip(".").lower()

        return f"{components.audio_stem}_{components.task.value}_{timestamp_str}.{clean_extension}"


class FilenameGenerator:
    """Context class that uses a FilenameGenerationStrategy to create filenames.
    Uses centralized timezone configuration from constants.py (APP_TIMEZONE
    environment variable, defaults to UTC) for generating timestamps
    or interpreting naive provided timestamps.
    """

    def __init__(self, strategy: FilenameGenerationStrategy):
        if not isinstance(strategy, FilenameGenerationStrategy):
            raise TypeError(
                "strategy must be an instance of FilenameGenerationStrategy"
            )
        self._strategy = strategy

    def create_filename(
        self,
        audio_path: str,
        task: TaskType,
        extension: str,
        timestamp: datetime.datetime | None = None,
    ) -> str:
        """Creates a filename for a given audio path, task, and extension.
        Uses centralized timezone configuration from constants.py
        (TZ environment variable via APP_TIMEZONE, defaults to Europe/Amsterdam).

        Args:
            audio_path: The full path to the original audio file.
            task: The type of ASR task (e.g., transcribe, translate).
            extension: The desired file extension for the output (e.g., "json", ".txt").
            timestamp: Optional. The specific timestamp to use.
                       - If None, the current time in the configured timezone will be used.
                       - If naive (no tzinfo), it's assumed to be in the configured timezone.
                       - If aware, it will be converted to the configured timezone.

        Returns:
            The generated filename string.
        """
        target_tz_str = APP_TIMEZONE
        try:
            target_tz = ZoneInfo(target_tz_str)
        except ZoneInfoNotFoundError:
            # Fallback to UTC if the configured timezone is invalid, and log a warning.
            # In a real application, this might raise an error or have more robust handling.
            print(
                f"Warning: Invalid timezone '{target_tz_str}' specified via TZ/APP_TIMEZONE. Falling back to UTC."
            )
            target_tz = ZoneInfo("UTC")

        if timestamp is None:
            # Generate current time in the target timezone
            timestamp = datetime.datetime.now(target_tz)
        else:
            if timestamp.tzinfo is None:
                # If timestamp is naive, assume it's in the target timezone
                timestamp = timestamp.replace(tzinfo=target_tz)
            else:
                # If timestamp is aware, convert it to the target timezone
                timestamp = timestamp.astimezone(target_tz)

        # Audio filename preservation (stem extraction)
        audio_basename = os.path.basename(audio_path)
        audio_stem = os.path.splitext(audio_basename)[0]

        components = FilenameComponents(
            audio_stem=audio_stem, task=task, timestamp=timestamp, extension=extension
        )
        return self._strategy.generate_filename(components)


# Example Usage (can be removed or moved to tests later):
# if __name__ == "__main__":
#     standard_strategy = StandardFilenameStrategy()
#     generator = FilenameGenerator(strategy=standard_strategy)

#     # Example 1: Basic transcription
#     filename1 = generator.create_filename(
#         audio_path="/path/to/my_audio_sample.wav",
#         task=TaskType.TRANSCRIBE,
#         extension="json"
#     )
#     print(f"Generated filename 1: {filename1}")
#     # Expected: my_audio_sample_transcribe_YYYYMMDDTHHMMSSZ.json

#     # Example 2: Translation with a specific timestamp
#     specific_time = datetime.datetime(2023, 10, 26, 12, 30, 0, tzinfo=datetime.timezone.utc)
#     filename2 = generator.create_filename(
#         audio_path="another_audio.mp3",
#         task=TaskType.TRANSLATE,
#         extension=".txt",
#         timestamp=specific_time
#     )
#     print(f"Generated filename 2: {filename2}")
#     # Expected: another_audio_translate_20231026T123000Z.txt

#     # Example 3: Testing stem extraction with multiple dots
#     filename3 = generator.create_filename(
#         audio_path="./recordings/my.meeting.notes.v2.m4a",
#         task=TaskType.TRANSCRIBE,
#         extension="srt"
#     )
#     print(f"Generated filename 3: {filename3}")
#     # Expected: my.meeting.notes.v2_transcribe_YYYYMMDDTHHMMSSZ.srt
