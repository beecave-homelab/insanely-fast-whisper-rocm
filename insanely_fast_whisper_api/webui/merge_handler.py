"""Merge Handler for WebUI Multiple File Processing.

This module implements the Template Method Pattern for merging multiple
transcription results into single files, with format-specific implementations
for TXT, SRT, and VTT formats.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insanely_fast_whisper_api.core.formatters import FORMATTERS

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.merge_handler")


@dataclass
class MergeConfiguration:
    """Configuration for file merging operations."""

    include_file_headers: bool = True
    include_section_separators: bool = True
    header_style: str = "equals"  # "equals", "hash", "dashes"
    section_separator: str = "\n\n"


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    merged_content: str
    source_files: list[str]
    format_type: str
    merge_stats: dict[str, Any]
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


class MergeHandler(ABC):
    """Abstract base class for format-specific merge handlers."""

    def __init__(self, config: MergeConfiguration | None = None) -> None:
        """Initialize the merge handler.

        Args:
            config: Optional merge configuration. If not provided, a default
                configuration is used.
        """
        self.config = config or MergeConfiguration()
        self.warnings = []

    def merge_files(self, file_results: dict[str, dict[str, Any]]) -> MergeResult:
        """Template method for merging multiple files.

        Args:
            file_results: Mapping of file path to transcription result data.

        Returns:
            MergeResult: Aggregate result with merged content and statistics.
        """
        try:
            logger.info(
                "Starting merge of %d files with %s",
                len(file_results),
                self.__class__.__name__,
            )

            validated_files = self._validate_files(file_results)
            if not validated_files:
                return MergeResult(
                    False,
                    "",
                    list(file_results.keys()),
                    self.get_format_name(),
                    {},
                    "No valid files",
                )

            ordered_files = sorted(
                validated_files.items(), key=lambda x: Path(x[0]).name.lower()
            )
            sections = self._format_sections(ordered_files)
            merged_content = self._combine_sections(sections)
            final_content = self._finalize_content(merged_content)

            stats = {
                "files_merged": len(validated_files),
                "format_type": self.get_format_name(),
            }

            return MergeResult(
                True,
                final_content,
                list(validated_files.keys()),
                self.get_format_name(),
                stats,
                warnings=self.warnings,
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            error_msg = f"Error merging files: {str(e)}"
            logger.error(error_msg)
            return MergeResult(
                False,
                "",
                list(file_results.keys()),
                self.get_format_name(),
                {},
                error_msg,
            )

    def _validate_files(
        self, file_results: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Validate input files.

        Args:
            file_results: Mapping of file path to raw result data.

        Returns:
            dict[str, dict[str, Any]]: Filtered mapping of valid files only.
        """
        validated = {}
        for file_path, result_data in file_results.items():
            if result_data and self._is_valid_file_result(result_data):
                validated[file_path] = result_data
            else:
                self.warnings.append(f"Invalid data for {Path(file_path).name}")
        return validated

    def _format_sections(
        self, ordered_files: list[tuple[str, dict[str, Any]]]
    ) -> list[str]:
        """Format each file as a section.

        Args:
            ordered_files: List of (path, result) tuples sorted deterministically.

        Returns:
            list[str]: A list of formatted section strings.
        """
        sections = []
        for file_path, result_data in ordered_files:
            try:
                section = ""
                if self.config.include_file_headers:
                    header = self._generate_header(file_path)
                    section += header + "\n"

                content = self._format_file_content(result_data)
                section += content
                sections.append(section)
            except (ValueError, TypeError, KeyError, IndexError) as e:
                self.warnings.append(
                    f"Error formatting {Path(file_path).name}: {str(e)}"
                )
        return sections

    def _combine_sections(self, sections: list[str]) -> str:
        """Combine sections.

        Args:
            sections: List of section strings.

        Returns:
            str: Combined content with optional separators.
        """
        if not sections:
            return ""
        separator = (
            self.config.section_separator
            if self.config.include_section_separators
            else "\n"
        )
        return separator.join(sections)

    def _finalize_content(self, content: str) -> str:
        """Finalize content.

        Args:
            content: The merged content.

        Returns:
            str: Final content possibly with headers/footers applied.
        """
        return content

    def _generate_header(self, file_path: str) -> str:
        """Generate section header.

        Args:
            file_path: Original file path.

        Returns:
            str: A formatted header string for the section.
        """
        filename = Path(file_path).name
        if self.config.header_style == "equals":
            return f"=== {filename} ==="
        elif self.config.header_style == "hash":
            return f"# {filename}"
        else:
            return f"--- {filename} ---"

    @abstractmethod
    def _is_valid_file_result(self, result_data: dict[str, Any]) -> bool:
        """Check if file result is valid."""
        raise NotImplementedError

    @abstractmethod
    def _format_file_content(self, result_data: dict[str, Any]) -> str:
        """Format file content."""
        raise NotImplementedError

    @abstractmethod
    def get_format_name(self) -> str:
        """Get format name.

        Returns:
            str: The format name handled by this merger.
        """
        raise NotImplementedError


class TxtMerger(MergeHandler):
    """Merge handler for TXT format."""

    def _is_valid_file_result(self, result_data: dict[str, Any]) -> bool:
        return "text" in result_data and result_data["text"].strip()

    def _format_file_content(self, result_data: dict[str, Any]) -> str:
        return FORMATTERS["txt"].format(result_data).strip()

    def get_format_name(self) -> str:
        """Return the human-readable format name handled by this merger.

        Returns:
            str: The format name "txt".
        """
        return "txt"


class SrtMerger(MergeHandler):
    """Merge handler for SRT format."""

    def __init__(self, config: MergeConfiguration | None = None) -> None:
        """Initialize the SRT merger.

        Args:
            config: Optional merge configuration.
        """
        super().__init__(config)
        self.entry_counter = 1

    def _is_valid_file_result(self, result_data: dict[str, Any]) -> bool:
        return ("chunks" in result_data and result_data["chunks"]) or (
            "segments" in result_data and result_data["segments"]
        )

    def _format_file_content(self, result_data: dict[str, Any]) -> str:
        srt_content = FORMATTERS["srt"].format(result_data)
        return self._renumber_srt(srt_content)

    def _renumber_srt(self, content: str) -> str:
        """Renumber SRT entries.

        Args:
            content: Original SRT content string.

        Returns:
            str: Content with sequentially renumbered entries.
        """
        entries = content.strip().split("\n\n")
        renumbered = []
        for entry in entries:
            if entry.strip():
                lines = entry.strip().split("\n")
                if len(lines) >= 3:
                    lines[0] = str(self.entry_counter)
                    renumbered.append("\n".join(lines))
                    self.entry_counter += 1
        return "\n\n".join(renumbered)

    def _finalize_content(self, content: str) -> str:
        return content  # No headers for SRT

    def get_format_name(self) -> str:
        """Return the human-readable format name handled by this merger.

        Returns:
            str: The format name "srt".
        """
        return "srt"


class VttMerger(MergeHandler):
    """Merge handler for VTT format."""

    def _is_valid_file_result(self, result_data: dict[str, Any]) -> bool:
        return ("chunks" in result_data and result_data["chunks"]) or (
            "segments" in result_data and result_data["segments"]
        )

    def _format_file_content(self, result_data: dict[str, Any]) -> str:
        return FORMATTERS["vtt"].format(result_data)

    def _finalize_content(self, content: str) -> str:
        return f"WEBVTT\n\n{content}" if content.strip() else "WEBVTT\n"

    def get_format_name(self) -> str:
        """Return the human-readable format name handled by this merger.

        Returns:
            str: The format name "vtt".
        """
        return "vtt"


# Handlers registry
MERGE_HANDLERS = {
    "txt": TxtMerger,
    "srt": SrtMerger,
    "vtt": VttMerger,
}


def get_merge_handler(
    format_type: str, config: MergeConfiguration | None = None
) -> MergeHandler:
    """Get merge handler for format.

    Args:
        format_type: One of "txt", "srt", or "vtt".
        config: Optional merge configuration.

    Returns:
        MergeHandler: A handler instance for the requested format.

    Raises:
        ValueError: If the requested format type is unsupported.
    """
    handler_class = MERGE_HANDLERS.get(format_type)
    if not handler_class:
        available = list(MERGE_HANDLERS.keys())
        raise ValueError(f"Unsupported format: {format_type}. Available: {available}")
    return handler_class(config)


def merge_files(
    file_results: dict[str, dict[str, Any]], format_type: str
) -> MergeResult:
    """Convenience function to merge files.

    Args:
        file_results: Mapping of file paths to result data.
        format_type: Target format for merge ("txt", "srt", or "vtt").

    Returns:
        MergeResult: The result including merged content and stats.
    """
    merger = get_merge_handler(format_type)
    return merger.merge_files(file_results)
