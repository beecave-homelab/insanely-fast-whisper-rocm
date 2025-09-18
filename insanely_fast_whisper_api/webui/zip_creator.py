"""ZIP Creator for WebUI Multiple File Processing.

This module implements the Builder Pattern for incrementally creating ZIP archives
with organized folder structures for batch transcription results, including
individual files by format, merged files, and batch summaries.
"""

import json
import logging
import os
import re
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from insanely_fast_whisper_api.core.formatters import FORMATTERS

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.zip_creator")


@dataclass
class ZipConfiguration:
    """Configuration for ZIP archive creation."""

    compression_method: int = zipfile.ZIP_DEFLATED
    compression_level: int | None = None  # None for default
    include_summary: bool = True
    include_merged: bool = False
    organize_by_format: bool = True
    organize_by_file: bool = False
    custom_structure: dict[str, str] | None = None  # format -> folder_name
    max_file_size_mb: int = 100  # Warning threshold
    temp_dir: str | None = None


@dataclass
class ZipStats:
    """Statistics for ZIP creation process."""

    files_added: int = 0
    folders_created: set[str] = field(default_factory=set)
    total_size_bytes: int = 0
    compression_ratio: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BatchZipBuilder:
    """Builder for creating organized ZIP archives of batch transcription results.

    Uses Builder Pattern to incrementally construct ZIP archives with flexible
    organization strategies and comprehensive error handling.
    """

    def __init__(self, config: ZipConfiguration | None = None) -> None:
        """Initialize the ZIP builder.

        Args:
            config: ZIP configuration options.
        """
        self.config = config or ZipConfiguration()
        self.stats = ZipStats()

        # Builder state
        self._zip_path: str | None = None
        self._zipfile: zipfile.ZipFile | None = None
        self._is_open = False
        self._batch_id: str | None = None
        self._batch_data: dict[str, Any] = {}

        # Content tracking
        self._individual_files: dict[
            str, dict[str, Any]
        ] = {}  # file_path -> result_data
        self._merged_content: dict[str, str] = {}  # format -> merged_content
        self._custom_files: list[tuple] = []  # [(archive_path, content)]

        logger.debug("Initialized BatchZipBuilder with config: %s", self.config)

    def create(
        self, batch_id: str | None = None, filename: str | None = None
    ) -> "BatchZipBuilder":
        """Create a new ZIP archive.

        Args:
            batch_id: Batch identifier for naming.
            filename: Custom filename for the archive.

        Returns:
            BatchZipBuilder: Self for method chaining.

        Raises:
            RuntimeError: If a ZIP archive is already open.
        """
        if self._is_open:
            raise RuntimeError("ZIP archive is already open")

        self._batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if filename:
            if not filename.endswith(".zip"):
                filename += ".zip"
        else:
            filename = f"batch_archive_{self._batch_id}.zip"

        temp_dir = self.config.temp_dir or tempfile.gettempdir()
        self._zip_path = os.path.join(temp_dir, filename)

        try:
            self._zipfile = zipfile.ZipFile(
                self._zip_path,
                "w",
                compression=self.config.compression_method,
                compresslevel=self.config.compression_level,
            )
            self._is_open = True

            logger.info("Created ZIP archive: %s", self._zip_path)
            return self

        except Exception as e:
            error_msg = "Failed to create ZIP archive: %s"
            logger.error(error_msg, str(e))
            self.stats.errors.append(error_msg % str(e))
            raise

    def add_batch_files(
        self, file_results: dict[str, dict[str, Any]], formats: list[str]
    ) -> "BatchZipBuilder":
        """Add batch transcription files to the archive.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.
            formats: List of formats to include.

        Returns:
            BatchZipBuilder: Self for method chaining.

        Raises:
            RuntimeError: If the ZIP archive is not open.
        """
        if not self._is_open:
            raise RuntimeError("ZIP archive is not open")
        assert self._zipfile is not None, "ZIP file not initialized"

        self._individual_files.update(file_results)

        try:
            if self.config.organize_by_format:
                self._add_files_by_format(file_results, formats)
            elif self.config.organize_by_file:
                self._add_files_by_source(file_results, formats)
            else:
                self._add_files_flat(file_results, formats)

            logger.info(
                "Added %d batch files in %d formats", len(file_results), len(formats)
            )
            return self

        except Exception as e:
            error_msg = "Failed to add batch files: %s"
            logger.error(error_msg, str(e))
            self.stats.errors.append(error_msg % str(e))
            raise

    def add_merged_files(
        self,
        file_results: dict[str, dict[str, Any]],
        formats: list[str],
        merged_filename: str | None = None,
    ) -> "BatchZipBuilder":
        """Add merged transcription files to the archive.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.
            formats: List of formats to merge.
            merged_filename: Custom filename base for merged files.

        Returns:
            BatchZipBuilder: Self for method chaining.

        Raises:
            RuntimeError: If the ZIP archive is not open.
            OSError | ValueError | TypeError | KeyError | AttributeError |
                zipfile.BadZipFile: On formatting or write failures.
        """
        if not self._is_open:
            raise RuntimeError("ZIP archive is not open")
        assert self._zipfile is not None, "ZIP file not initialized"

        try:
            merged_base = merged_filename or f"batch_merged_{self._batch_id}"

            for format_type in formats:
                merged_content = self._merge_format(file_results, format_type)
                self._merged_content[format_type] = merged_content

                extension = FORMATTERS[format_type].get_file_extension()
                archive_path = f"merged/{merged_base}.{extension}"

                self._zipfile.writestr(archive_path, merged_content)
                self._track_addition(archive_path, len(merged_content.encode("utf-8")))

                logger.debug("Added merged %s file: %s", format_type, archive_path)

            logger.info("Added %d merged files", len(formats))
            return self

        except Exception as e:
            error_msg = "Failed to add merged files: %s"
            logger.error(error_msg, str(e))
            self.stats.errors.append(error_msg % str(e))
            raise

    def add_custom_file(self, archive_path: str, content: str) -> "BatchZipBuilder":
        """Add a custom file to the archive.

        Args:
            archive_path: Path within the archive.
            content: File content as string.

        Returns:
            BatchZipBuilder: Self for method chaining.

        Raises:
            RuntimeError: If the ZIP archive is not open.
        """
        if not self._is_open:
            raise RuntimeError("ZIP archive is not open")
        assert self._zipfile is not None, "ZIP file not initialized"

        try:
            self._zipfile.writestr(archive_path, content)
            self._track_addition(archive_path, len(content.encode("utf-8")))
            self._custom_files.append((archive_path, content))

            logger.debug("Added custom file: %s", archive_path)
            return self

        except Exception as e:
            error_msg = "Failed to add custom file %s: %s"
            logger.error(error_msg, archive_path, str(e))
            self.stats.errors.append(error_msg % (archive_path, str(e)))
            raise

    def add_summary(self, include_stats: bool = True) -> "BatchZipBuilder":
        """Add a batch summary file to the archive.

        Args:
            include_stats: Whether to include ZIP creation statistics.

        Returns:
            BatchZipBuilder: Self for method chaining.

        Raises:
            RuntimeError: If the ZIP archive is not open.
        """
        if not self._is_open:
            raise RuntimeError("ZIP archive is not open")
        assert self._zipfile is not None, "ZIP file not initialized"

        if not self.config.include_summary:
            return self

        try:
            summary = self._generate_summary(include_stats)
            summary_content = json.dumps(summary, indent=2, ensure_ascii=False)

            self._zipfile.writestr("batch_summary.json", summary_content)
            self._track_addition(
                "batch_summary.json", len(summary_content.encode("utf-8"))
            )

            logger.info("Added batch summary file")
            return self

        except (OSError, ValueError, TypeError, zipfile.BadZipFile) as e:
            error_msg = "Failed to add summary: %s"
            logger.error(error_msg, str(e))
            self.stats.errors.append(error_msg % str(e))
            # Don't raise for summary failures
            return self

    def build(self) -> tuple[str, ZipStats]:
        """Finalize and close the ZIP archive.

        Returns:
            tuple[str, ZipStats]: A tuple of the ZIP path and creation statistics.

        Raises:
            RuntimeError: If the archive is not open or no path was created.
        """
        if not self._is_open:
            raise RuntimeError("ZIP archive is not open")

        try:
            # Add summary automatically if enabled
            if self.config.include_summary:
                self.add_summary(include_stats=True)

            # Calculate final compression ratio
            if self._zipfile:
                # Get uncompressed size from infolist
                uncompressed_size = sum(
                    info.file_size for info in self._zipfile.infolist()
                )
                compressed_size = sum(
                    info.compress_size for info in self._zipfile.infolist()
                )

                if uncompressed_size > 0:
                    self.stats.compression_ratio = (
                        1 - compressed_size / uncompressed_size
                    ) * 100

                # Close the zipfile here and mark as closed
                self._zipfile.close()
                self._zipfile = None
                self._is_open = False

            # Get final file size
            if self._zip_path and os.path.exists(self._zip_path):
                file_size_mb = os.path.getsize(self._zip_path) / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    warning = (
                        f"ZIP file size ({file_size_mb:.1f}MB) exceeds threshold "
                        f"({self.config.max_file_size_mb:.1f}MB)"
                    )
                    self.stats.warnings.append(warning)
                    logger.warning(warning)

            if self._zip_path is None:
                raise RuntimeError("ZIP archive path was not created.")

            logger.info(
                "Built ZIP archive: %s (%.1f%% compression, %d files)",
                self._zip_path,
                self.stats.compression_ratio,
                self.stats.files_added,
            )

            return self._zip_path, self.stats

        except Exception as e:
            error_msg = "Failed to build ZIP archive: %s"
            logger.error(error_msg, str(e))
            self.stats.errors.append(error_msg % str(e))
            raise
        finally:
            # Ensure zipfile is closed and state is reset, but only if not already done
            if self._zipfile:
                try:
                    self._zipfile.close()
                except OSError:
                    pass  # Ignore errors during cleanup
                self._zipfile = None
            self._is_open = False

    def _add_files_by_format(
        self, file_results: dict[str, dict[str, Any]], formats: list[str]
    ) -> None:
        """Add files organized by format (formats/txt/, formats/srt/, etc.)."""
        assert self._zipfile is not None, "ZIP file not initialized"
        for format_type in formats:
            formatter = FORMATTERS.get(format_type)
            if not formatter:
                logger.warning("No formatter found for %s, skipping.", format_type)
                continue

            format_folder = format_type.lower()
            extension = formatter.get_file_extension()

            for file_path, result_data in file_results.items():
                original_filename = self._get_base_filename(file_path)
                archive_filename = f"{original_filename}.{extension}"
                archive_path = f"{format_folder}/{archive_filename}"

                logger.debug(
                    "[_add_files_by_format] Preparing to format for: %s",
                    archive_path,
                )
                logger.debug(
                    "[_add_files_by_format] Raw result_data for %s: %s",
                    file_path,
                    json.dumps(result_data, indent=2),
                )

                content = self._format_result(result_data, format_type)

                # Log content snippet or length for brevity
                content_log = content[:200] + "..." if len(content) > 200 else content
                logger.debug(
                    ("[_add_files_by_format] Formatted content for %s (len %d): '%s'"),
                    archive_path,
                    len(content),
                    content_log,
                )

                if (
                    not content.strip() and format_type != "json"
                ):  # JSON can be {} which is valid
                    logger.warning(
                        "[_add_files_by_format] Formatter for %s produced empty or "
                        "whitespace-only content for %s. Writing empty file to ZIP.",
                        format_type,
                        file_path,
                    )
                elif (
                    not content.strip()
                    and format_type == "json"
                    and content not in ["{}", "null"]
                ):  # more specific for json
                    logger.warning(
                        (
                            "[_add_files_by_format] Formatter for JSON produced empty "
                            "or whitespace-only content for %s (actual: '%s'). "
                            "Writing to ZIP."
                        ),
                        file_path,
                        content,
                    )

                self._zipfile.writestr(archive_path, content)
                self._track_addition(archive_path, len(content.encode("utf-8")))

    def _add_files_by_source(
        self, file_results: dict[str, dict[str, Any]], formats: list[str]
    ) -> None:
        """Add files organized by source file (files/audio1/, files/audio2/, etc.)."""
        assert self._zipfile is not None, "ZIP file not initialized"
        for file_path, result_data in file_results.items():
            source_folder = Path(file_path).stem
            self.stats.folders_created.add(f"files/{source_folder}")

            for format_type in formats:
                try:
                    content = self._format_result(result_data, format_type)
                    extension = FORMATTERS[format_type].get_file_extension()

                    archive_path = f"files/{source_folder}/{source_folder}.{extension}"
                    self._zipfile.writestr(archive_path, content)
                    self._track_addition(archive_path, len(content.encode("utf-8")))

                except (
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    zipfile.BadZipFile,
                    UnicodeEncodeError,
                ) as e:
                    error_msg = "Failed to add %s file for %s: %s"
                    logger.error(error_msg, format_type, source_folder, str(e))
                    self.stats.errors.append(
                        error_msg % (format_type, source_folder, str(e))
                    )

    def _add_files_flat(
        self, file_results: dict[str, dict[str, Any]], formats: list[str]
    ) -> None:
        """Add files in flat structure (no subfolders)."""
        assert self._zipfile is not None, "ZIP file not initialized"
        for file_path, result_data in file_results.items():
            original_filename = self._get_base_filename(file_path)

            for format_type in formats:
                formatter = FORMATTERS.get(format_type)
                if not formatter:
                    logger.warning(
                        "No formatter found for %s, skipping for %s.",
                        format_type,
                        file_path,
                    )
                    continue

                extension = formatter.get_file_extension()
                archive_filename = f"{original_filename}.{extension}"
                # For flat structure, files are at the root
                archive_path = archive_filename

                logger.debug(
                    "[_add_files_flat] Preparing to format for: %s",
                    archive_path,
                )
                logger.debug(
                    "[_add_files_flat] Raw result_data for %s: %s",
                    file_path,
                    json.dumps(result_data, indent=2),
                )

                content = self._format_result(result_data, format_type)

                # Log content snippet or length for brevity
                content_log = content[:200] + "..." if len(content) > 200 else content
                logger.debug(
                    ("[_add_files_flat] Formatted content for %s (len %d): '%s'"),
                    archive_path,
                    len(content),
                    content_log,
                )

                if (
                    not content.strip() and format_type != "json"
                ):  # JSON can be {} which is valid
                    logger.warning(
                        "[_add_files_flat] Formatter for %s produced empty or "
                        "whitespace-only content for %s. Writing empty file to ZIP.",
                        format_type,
                        file_path,
                    )
                elif (
                    not content.strip()
                    and format_type == "json"
                    and content not in ["{}", "null"]
                ):  # more specific for json
                    logger.warning(
                        (
                            "[_add_files_flat] Formatter for JSON produced empty or "
                            "whitespace-only content for %s (actual: '%s'). "
                            "Writing to ZIP."
                        ),
                        file_path,
                        content,
                    )

                self._zipfile.writestr(archive_path, content)
                self._track_addition(archive_path, len(content.encode("utf-8")))

    def _merge_format(
        self, file_results: dict[str, dict[str, Any]], format_type: str
    ) -> str:
        """Merge multiple transcription results into a single format.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.
            format_type: The desired format to merge (e.g., "txt", "srt", "json").

        Returns:
            str: The merged content.

        Raises:
            ValueError: If an unsupported merge format is requested.
        """
        if format_type == "txt":
            return self._merge_txt(file_results)
        elif format_type == "srt":
            return self._merge_srt(file_results)
        elif format_type == "json":
            return self._merge_json(file_results)
        else:
            raise ValueError(f"Unsupported merge format: {format_type}")

    def _merge_txt(self, file_results: dict[str, dict[str, Any]]) -> str:
        """Merge TXT format files.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.

        Returns:
            str: Merged plain text content with headers and separators.
        """
        merged_lines = []

        for file_path, result_data in file_results.items():
            filename = Path(file_path).name
            text = result_data.get("text", "").strip()

            if text:
                merged_lines.append(f"=== {filename} ===")
                merged_lines.append(text)
                merged_lines.append("")  # Empty line separator

        return "\n".join(merged_lines)

    def _merge_srt(self, file_results: dict[str, dict[str, Any]]) -> str:
        """Merge SRT format files with sequential numbering.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.

        Returns:
            str: Merged SRT content with renumbered entries and section comments.
        """
        merged_entries = []
        entry_counter = 1

        for file_path, result_data in file_results.items():
            filename = Path(file_path).name
            srt_content = self._format_result(result_data, "srt")

            if srt_content.strip():
                # Add file header comment
                merged_entries.append(f"# Source: {filename}")

                # Parse and renumber SRT entries
                entries = srt_content.strip().split("\n\n")
                for entry in entries:
                    if entry.strip():
                        lines = entry.strip().split("\n")
                        if len(lines) >= 3:
                            # Replace entry number
                            lines[0] = str(entry_counter)
                            merged_entries.append("\n".join(lines))
                            entry_counter += 1

                merged_entries.append("")  # Section separator

        return "\n\n".join(filter(None, merged_entries))

    def _merge_json(self, file_results: dict[str, dict[str, Any]]) -> str:
        """Merge JSON format files into a batch structure.

        Args:
            file_results: Dictionary of ``file_path -> transcription_result``.

        Returns:
            str: JSON string representing batch info and per-file results.
        """
        files_data: dict[str, dict[str, Any]] = {}
        for file_path, result_data in file_results.items():
            filename = Path(file_path).name
            files_data[filename] = result_data

        batch_data: dict[str, Any] = {
            "batch_info": {
                "batch_id": self._batch_id,
                "total_files": len(file_results),
                "created_timestamp": datetime.now().isoformat(),
            },
            "files": files_data,
        }
        return json.dumps(batch_data, indent=2, ensure_ascii=False)

    def _format_result(self, result_data: dict[str, Any], format_type: str) -> str:
        """Format transcription result data.

        Args:
            result_data: Raw transcription result data.
            format_type: One of "txt", "srt", or "json".

        Returns:
            str: The formatted content.

        Raises:
            ValueError: If the requested ``format_type`` is unknown.
        """
        formatter = FORMATTERS.get(format_type)
        if not formatter:
            raise ValueError(f"Unknown format: {format_type}")
        return formatter.format(result_data)

    def _get_base_filename(self, file_path: str) -> str:
        """Get base filename from path with safe Unicode normalization.

        Args:
            file_path: The original file path.

        Returns:
            str: A sanitized base filename safe for archive inclusion.
        """
        base_name = Path(file_path).stem

        # Normalize Unicode characters (NFC normalization)
        normalized_name = unicodedata.normalize("NFC", base_name)

        # Conservative sanitization - only remove characters that are genuinely
        # problematic across all filesystems. Keep parentheses (), brackets [],
        # spaces, hyphens, etc. Only remove: < > : " | ? * (Windows restrictions)
        # and control characters (\x00-\x1f)
        safe_name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", normalized_name)

        # Only trim leading/trailing whitespace (keep internal spaces)
        safe_name = safe_name.strip()

        # Remove leading/trailing dots only (Windows compatibility)
        safe_name = safe_name.strip(".")

        # Ensure the filename isn't empty after cleaning
        if not safe_name:
            safe_name = "file"

        # Use reasonable length limit for ZIP archives (ZIP standard supports much more)
        max_name_length = 100
        if len(safe_name) > max_name_length:
            safe_name = safe_name[:max_name_length].strip(" .")

        return safe_name

    def _track_addition(self, archive_path: str, size_bytes: int) -> None:
        """Track file addition statistics.

        Args:
            archive_path: Archive path of the added file.
            size_bytes: Size of the added content in bytes.
        """
        self.stats.files_added += 1
        self.stats.total_size_bytes += size_bytes

        # Track folder creation
        folder = str(Path(archive_path).parent)
        if folder != ".":
            self.stats.folders_created.add(folder)

    def _generate_summary(self, include_stats: bool) -> dict[str, Any]:
        """Generate batch summary data.

        Args:
            include_stats: Whether to include creation statistics.

        Returns:
            dict[str, Any]: Summary structure for the batch ZIP.
        """
        summary: dict[str, Any] = {
            "batch_info": {
                "batch_id": self._batch_id,
                "created_timestamp": datetime.now().isoformat(),
                "total_individual_files": len(self._individual_files),
                "formats_included": (
                    list(self._merged_content.keys()) if self._merged_content else []
                ),
                "organization_strategy": (
                    "by_format"
                    if self.config.organize_by_format
                    else "by_file"
                    if self.config.organize_by_file
                    else "flat"
                ),
                "archive_structure": {
                    "formats/": (
                        "Individual files organized by format"
                        if self.config.organize_by_format
                        else None
                    ),
                    "files/": (
                        "Individual files organized by source"
                        if self.config.organize_by_file
                        else None
                    ),
                    "merged/": "Merged files" if self._merged_content else None,
                    "batch_summary.json": "This summary file",
                },
            },
            "file_details": [
                {
                    "file_path": file_path,
                    "filename": Path(file_path).name,
                    "status": "included",
                }
                for file_path in self._individual_files.keys()
            ],
        }

        # Remove None values from structure
        structure_dict = summary["batch_info"]["archive_structure"]
        if isinstance(structure_dict, dict):
            summary["batch_info"]["archive_structure"] = {
                k: v for k, v in structure_dict.items() if v is not None
            }

        if include_stats:
            summary["creation_stats"] = {
                "files_added": self.stats.files_added,
                "folders_created": sorted(list(self.stats.folders_created)),
                "total_size_bytes": self.stats.total_size_bytes,
                "compression_ratio_percent": round(self.stats.compression_ratio, 2),
                "errors": self.stats.errors,
                "warnings": self.stats.warnings,
            }

        return summary

    def __enter__(self) -> "BatchZipBuilder":
        """Context manager entry.

        Returns:
            BatchZipBuilder: Self for use within a context manager.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit.

        Args:
            exc_type: Exception type if one occurred, otherwise None.
            exc_val: Exception instance if one occurred, otherwise None.
            exc_tb: Traceback if one occurred, otherwise None.
        """
        if self._is_open and self._zipfile:
            try:
                self._zipfile.close()
            except OSError as e:
                logger.error("Error closing ZIP file: %s", str(e))
            finally:
                self._is_open = False


# Convenience function for simple ZIP creation
def create_batch_zip(
    file_results: dict[str, dict[str, Any]],
    formats: list[str],
    batch_id: str | None = None,
    include_merged: bool = False,
    config: ZipConfiguration | None = None,
) -> tuple[str, ZipStats]:
    """Create a ZIP archive for batch transcription results.

    Args:
        file_results: Dictionary of ``file_path -> transcription_result``.
        formats: List of formats to include.
        batch_id: Optional batch identifier.
        include_merged: Whether to include merged files.
        config: ZIP configuration.

    Returns:
        tuple[str, ZipStats]: Tuple of (zip_path, zip_stats).
    """
    builder = BatchZipBuilder(config)

    with builder:
        builder.create(batch_id)
        builder.add_batch_files(file_results, formats)

        if include_merged:
            builder.add_merged_files(file_results, formats)

        return builder.build()
