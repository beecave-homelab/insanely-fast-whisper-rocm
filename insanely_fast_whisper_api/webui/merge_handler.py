"""Merge Handler for WebUI Multiple File Processing.

This module implements the Template Method Pattern for merging multiple
transcription results into single files, with format-specific implementations
for TXT, SRT, and VTT formats.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from insanely_fast_whisper_api.webui.formatters import FORMATTERS

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
    source_files: List[str]
    format_type: str
    merge_stats: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class MergeHandler(ABC):
    """Abstract base class for format-specific merge handlers."""

    def __init__(self, config: Optional[MergeConfiguration] = None):
        self.config = config or MergeConfiguration()
        self.warnings = []

    def merge_files(self, file_results: Dict[str, Dict[str, Any]]) -> MergeResult:
        """Template method for merging multiple files."""
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

        except (IOError, OSError, ValueError, TypeError, KeyError, AttributeError) as e:
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
        self, file_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate input files."""
        validated = {}
        for file_path, result_data in file_results.items():
            if result_data and self._is_valid_file_result(result_data):
                validated[file_path] = result_data
            else:
                self.warnings.append(f"Invalid data for {Path(file_path).name}")
        return validated

    def _format_sections(
        self, ordered_files: List[Tuple[str, Dict[str, Any]]]
    ) -> List[str]:
        """Format each file as a section."""
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

    def _combine_sections(self, sections: List[str]) -> str:
        """Combine sections."""
        if not sections:
            return ""
        separator = (
            self.config.section_separator
            if self.config.include_section_separators
            else "\n"
        )
        return separator.join(sections)

    def _finalize_content(self, content: str) -> str:
        """Finalize content."""
        return content

    def _generate_header(self, file_path: str) -> str:
        """Generate section header."""
        filename = Path(file_path).name
        if self.config.header_style == "equals":
            return f"=== {filename} ==="
        elif self.config.header_style == "hash":
            return f"# {filename}"
        else:
            return f"--- {filename} ---"

    @abstractmethod
    def _is_valid_file_result(self, result_data: Dict[str, Any]) -> bool:
        """Check if file result is valid."""
        raise NotImplementedError

    @abstractmethod
    def _format_file_content(self, result_data: Dict[str, Any]) -> str:
        """Format file content."""
        raise NotImplementedError

    @abstractmethod
    def get_format_name(self) -> str:
        """Get format name."""
        raise NotImplementedError


class TxtMerger(MergeHandler):
    """Merge handler for TXT format."""

    def _is_valid_file_result(self, result_data: Dict[str, Any]) -> bool:
        return "text" in result_data and result_data["text"].strip()

    def _format_file_content(self, result_data: Dict[str, Any]) -> str:
        return FORMATTERS["txt"].format(result_data).strip()

    def get_format_name(self) -> str:
        return "txt"


class SrtMerger(MergeHandler):
    """Merge handler for SRT format."""

    def __init__(self, config: Optional[MergeConfiguration] = None):
        super().__init__(config)
        self.entry_counter = 1

    def _is_valid_file_result(self, result_data: Dict[str, Any]) -> bool:
        return "chunks" in result_data and result_data["chunks"]

    def _format_file_content(self, result_data: Dict[str, Any]) -> str:
        srt_content = FORMATTERS["srt"].format(result_data)
        return self._renumber_srt(srt_content)

    def _renumber_srt(self, content: str) -> str:
        """Renumber SRT entries."""
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
        return "srt"


class VttMerger(MergeHandler):
    """Merge handler for VTT format."""

    def _is_valid_file_result(self, result_data: Dict[str, Any]) -> bool:
        return "chunks" in result_data and result_data["chunks"]

    def _format_file_content(self, result_data: Dict[str, Any]) -> str:
        chunks = result_data.get("chunks", [])
        vtt_entries = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            timestamps = chunk.get("timestamp", [None, None])
            start = timestamps[0] if len(timestamps) > 0 else None
            end = timestamps[1] if len(timestamps) > 1 else None

            start_time = self._format_vtt_time(start)
            end_time = self._format_vtt_time(end)

            vtt_entries.append(f"{start_time} --> {end_time}\n{text}")

        return "\n\n".join(vtt_entries)

    def _format_vtt_time(self, seconds: Optional[float]) -> str:
        """Format time for VTT."""
        if seconds is None:
            return "00:00:00.000"

        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        secs = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def _finalize_content(self, content: str) -> str:
        return f"WEBVTT\n\n{content}" if content.strip() else "WEBVTT\n"

    def get_format_name(self) -> str:
        return "vtt"


# Handlers registry
MERGE_HANDLERS = {
    "txt": TxtMerger,
    "srt": SrtMerger,
    "vtt": VttMerger,
}


def get_merge_handler(
    format_type: str, config: Optional[MergeConfiguration] = None
) -> MergeHandler:
    """Get merge handler for format."""
    handler_class = MERGE_HANDLERS.get(format_type)
    if not handler_class:
        available = list(MERGE_HANDLERS.keys())
        raise ValueError(f"Unsupported format: {format_type}. Available: {available}")
    return handler_class(config)


def merge_files(
    file_results: Dict[str, Dict[str, Any]], format_type: str
) -> MergeResult:
    """Convenience function to merge files."""
    merger = get_merge_handler(format_type)
    return merger.merge_files(file_results)
