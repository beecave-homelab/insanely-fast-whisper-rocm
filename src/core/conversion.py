"""
Core module for handling transcription format conversions.

This module provides functionality to convert transcription results between
various formats such as JSON, TXT, SRT, and VTT.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Type, Any
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator


class OutputFormat(str, Enum):
    """Supported output formats for transcription conversion."""

    JSON = "json"
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"


class BaseFormatter:
    """Base class for all formatters."""

    @classmethod
    def get_extension(cls) -> str:
        """Get the file extension for this format."""
        raise NotImplementedError

    @classmethod
    def get_content_type(cls) -> str:
        """Get the MIME content type for this format."""
        raise NotImplementedError

    @classmethod
    def format_timestamp(cls, seconds: Optional[float]) -> str:
        """Format a timestamp in seconds to the format required by the formatter.

        Args:
            seconds: Timestamp in seconds, or None.

        Returns:
            Formatted timestamp string.
        """
        raise NotImplementedError

    @classmethod
    def format_chunk(
        cls, text: str, start: Optional[float], end: Optional[float], index: int
    ) -> str:
        """Format a single chunk of transcribed text.

        Args:
            text: The transcribed text.
            start: Start time in seconds.
            end: End time in seconds.
            index: Chunk index (1-based).

        Returns:
            Formatted chunk as a string.
        """
        raise NotImplementedError

    @classmethod
    def format_document(
        cls, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a complete document from transcription chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'timestamp' keys.
            metadata: Optional metadata to include in the output.

        Returns:
            Complete formatted document as a string.
        """
        raise NotImplementedError


class JSONFormatter(BaseFormatter):
    """Formatter for JSON output."""

    @classmethod
    def get_extension(cls) -> str:
        return "json"

    @classmethod
    def get_content_type(cls) -> str:
        return "application/json"

    @classmethod
    def format_timestamp(cls, seconds: Optional[float]) -> str:
        return str(seconds) if seconds is not None else ""

    @classmethod
    def format_chunk(
        cls, text: str, start: Optional[float], end: Optional[float], index: int
    ) -> Dict[str, Any]:
        return {"index": index, "text": text, "start": start, "end": end}

    @classmethod
    def format_document(
        cls, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        result = {
            "metadata": metadata or {},
            "chunks": [
                {
                    "text": chunk.get("text", ""),
                    "start": (
                        chunk.get("timestamp", [None])[0]
                        if isinstance(chunk.get("timestamp"), list)
                        else None
                    ),
                    "end": (
                        chunk.get("timestamp", [None, None])[1]
                        if isinstance(chunk.get("timestamp"), list)
                        else None
                    ),
                }
                for chunk in chunks
            ],
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


class TXTFormatter(BaseFormatter):
    """Formatter for plain text output."""

    @classmethod
    def get_extension(cls) -> str:
        return "txt"

    @classmethod
    def get_content_type(cls) -> str:
        return "text/plain"

    @classmethod
    def format_timestamp(cls, seconds: Optional[float]) -> str:
        return ""

    @classmethod
    def format_chunk(
        cls, text: str, start: Optional[float], end: Optional[float], index: int
    ) -> str:
        return f"{text}\n"

    @classmethod
    def format_document(
        cls, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        return "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))


class SRTFormatter(BaseFormatter):
    """Formatter for SRT (SubRip) subtitle format."""

    @classmethod
    def get_extension(cls) -> str:
        return "srt"

    @classmethod
    def get_content_type(cls) -> str:
        return "application/x-subrip"

    @classmethod
    def format_timestamp(cls, seconds: Optional[float]) -> str:
        if seconds is None:
            return "00:00:00,000"

        try:
            whole_seconds = int(seconds)
            milliseconds = int((seconds - whole_seconds) * 1000)

            hours = whole_seconds // 3600
            minutes = (whole_seconds % 3600) // 60
            seconds = whole_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        except (TypeError, ValueError):
            return "00:00:00,000"

    @classmethod
    def format_chunk(
        cls, text: str, start: Optional[float], end: Optional[float], index: int
    ) -> str:
        start_fmt = cls.format_timestamp(start)
        end_fmt = cls.format_timestamp(end)
        return f"{index}\n{start_fmt} --> {end_fmt}\n{text}\n"

    @classmethod
    def format_document(
        cls, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        result = []
        for i, chunk in enumerate(chunks, 1):
            if not chunk.get("text"):
                continue

            start, end = chunk.get("timestamp", [None, None])
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue

            result.append(cls.format_chunk(chunk["text"], start, end, i))

        return "\n".join(result).strip()


class VTTFormatter(BaseFormatter):
    """Formatter for WebVTT subtitle format."""

    @classmethod
    def get_extension(cls) -> str:
        return "vtt"

    @classmethod
    def get_content_type(cls) -> str:
        return "text/vtt"

    @classmethod
    def format_timestamp(cls, seconds: Optional[float]) -> str:
        if seconds is None:
            return "00:00:00.000"

        try:
            whole_seconds = int(seconds)
            milliseconds = int((seconds - whole_seconds) * 1000)

            hours = whole_seconds // 3600
            minutes = (whole_seconds % 3600) // 60
            seconds = whole_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        except (TypeError, ValueError):
            return "00:00:00.000"

    @classmethod
    def format_chunk(
        cls, text: str, start: Optional[float], end: Optional[float], index: int
    ) -> str:
        start_fmt = cls.format_timestamp(start)
        end_fmt = cls.format_timestamp(end)
        return f"{start_fmt} --> {end_fmt}\n{text}\n"

    @classmethod
    def format_document(
        cls, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        result = ["WEBVTT", ""]

        for i, chunk in enumerate(chunks, 1):
            if not chunk.get("text"):
                continue

            start, end = chunk.get("timestamp", [None, None])
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue

            result.append(f"{i}")
            result.append(
                f"{cls.format_timestamp(start)} --> {cls.format_timestamp(end)}"
            )
            result.append(f"{chunk['text']}\n")

        return "\n".join(result).strip()


# Map of format names to formatter classes
FORMATTERS = {
    OutputFormat.JSON: JSONFormatter,
    OutputFormat.TXT: TXTFormatter,
    OutputFormat.SRT: SRTFormatter,
    OutputFormat.VTT: VTTFormatter,
}


def get_formatter(format_name: Union[str, OutputFormat]) -> Type[BaseFormatter]:
    """Get the appropriate formatter for the given format name.

    Args:
        format_name: Name or enum of the format.

    Returns:
        Formatter class for the specified format.

    Raises:
        ValueError: If the format is not supported.
    """
    if isinstance(format_name, str):
        format_name = OutputFormat(format_name.lower())

    formatter = FORMATTERS.get(format_name)
    if formatter is None:
        raise ValueError(
            f"Unsupported format: {format_name}. Supported formats: {list(FORMATTERS.keys())}"
        )

    return formatter


def convert_transcription(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    output_format: Union[str, OutputFormat] = OutputFormat.TXT,
    input_format: Optional[Union[str, OutputFormat]] = None,
) -> str:
    """Convert a transcription file from one format to another.

    Args:
        input_path: Path to the input file.
        output_path: Path to save the output file. If None, returns the result as a string.
        output_format: Desired output format.
        input_format: Input format. If None, inferred from file extension.

    Returns:
        The converted content as a string if output_path is None, otherwise an empty string.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input or output format is not supported.
        json.JSONDecodeError: If the input file is not valid JSON.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine input format if not specified
    if input_format is None:
        ext = input_path.suffix[1:].lower()
        try:
            input_format = OutputFormat(ext)
        except ValueError:
            raise ValueError(
                f"Could not determine input format from file extension. "
                f"Please specify input_format. Supported formats: {list(FORMATTERS.keys())}"
            )

    # Get the appropriate formatters
    output_formatter = get_formatter(output_format)

    # Read and parse the input file
    with open(input_path, "r", encoding="utf-8") as f:
        if input_format == OutputFormat.JSON:
            data = json.load(f)
            chunks = data.get("chunks", [])
            metadata = data.get("metadata", {})
        else:
            # For non-JSON input, we'd need to implement parsers for each format
            # For now, we'll just support JSON input
            raise ValueError(
                f"Input format {input_format} is not yet supported. Only JSON input is currently supported."
            )

    # Format the output
    output_content = output_formatter.format_document(chunks, metadata)

    # Write to file or return as string
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        return ""

    return output_content


def batch_convert_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: Union[str, OutputFormat] = OutputFormat.TXT,
    input_extension: str = "json",
    recursive: bool = False,
) -> Dict[str, Union[Path, Exception]]:
    """Convert all files in a directory from one format to another.

    Args:
        input_dir: Directory containing input files.
        output_dir: Directory to save output files.
        output_format: Desired output format.
        input_extension: File extension of input files (without leading dot).
        recursive: Whether to search for files recursively in subdirectories.

    Returns:
        Dictionary mapping input file paths to output file paths or exceptions.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    pattern = f"**/*.{input_extension}" if recursive else f"*.{input_extension}"
    input_files = list(input_dir.glob(pattern))

    results = {}

    for input_file in input_files:
        if not input_file.is_file():
            continue

        rel_path = input_file.relative_to(input_dir)
        output_file = (
            output_dir
            / f"{rel_path.with_suffix('.' + output_formatter.get_extension())}"
        )

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            convert_transcription(
                input_path=input_file,
                output_path=output_file,
                output_format=output_format,
                input_format=input_extension,
            )
            results[str(input_file)] = output_file
        except Exception as e:
            results[str(input_file)] = e
            logging.error(f"Failed to convert {input_file}: {str(e)}")

    return results
