"""Tests for insanely_fast_whisper_rocm.webui.errors module.

This module contains tests for custom exception classes used in the WebUI.
"""

from __future__ import annotations

import pytest

from insanely_fast_whisper_rocm.webui.errors import (
    DeviceNotFoundError,
    ExportError,
    FormatterError,
    TranscriptionError,
)


class TestTranscriptionError:
    """Test suite for TranscriptionError exception."""

    def test_transcription_error__can_be_raised(self) -> None:
        """Test that TranscriptionError can be raised.

        Raises:
            TranscriptionError: To verify it can be caught.
        """
        with pytest.raises(TranscriptionError):
            raise TranscriptionError("Test error")

    def test_transcription_error__with_message(self) -> None:
        """Test that TranscriptionError preserves error message.

        Raises:
            TranscriptionError: With a specific error message.
        """
        error_msg = "Transcription failed due to invalid audio"
        with pytest.raises(TranscriptionError, match=error_msg):
            raise TranscriptionError(error_msg)

    def test_transcription_error__is_exception(self) -> None:
        """Test that TranscriptionError is a subclass of Exception."""
        assert issubclass(TranscriptionError, Exception)

    def test_transcription_error__empty_message(self) -> None:
        """Test that TranscriptionError can be raised with empty message.

        Raises:
            TranscriptionError: With no message.
        """
        with pytest.raises(TranscriptionError):
            raise TranscriptionError()


class TestDeviceNotFoundError:
    """Test suite for DeviceNotFoundError exception."""

    def test_device_not_found_error__can_be_raised(self) -> None:
        """Test that DeviceNotFoundError can be raised.

        Raises:
            DeviceNotFoundError: To verify it can be caught.
        """
        with pytest.raises(DeviceNotFoundError):
            raise DeviceNotFoundError("Device not available")

    def test_device_not_found_error__with_message(self) -> None:
        """Test that DeviceNotFoundError preserves error message.

        Raises:
            DeviceNotFoundError: With a specific error message.
        """
        error_msg = "CUDA device not found"
        with pytest.raises(DeviceNotFoundError, match=error_msg):
            raise DeviceNotFoundError(error_msg)

    def test_device_not_found_error__is_exception(self) -> None:
        """Test that DeviceNotFoundError is a subclass of Exception."""
        assert issubclass(DeviceNotFoundError, Exception)

    def test_device_not_found_error__empty_message(self) -> None:
        """Test that DeviceNotFoundError can be raised with empty message.

        Raises:
            DeviceNotFoundError: With no message.
        """
        with pytest.raises(DeviceNotFoundError):
            raise DeviceNotFoundError()


class TestFormatterError:
    """Test suite for FormatterError exception."""

    def test_formatter_error__can_be_raised(self) -> None:
        """Test that FormatterError can be raised.

        Raises:
            FormatterError: To verify it can be caught.
        """
        with pytest.raises(FormatterError):
            raise FormatterError("Formatting failed")

    def test_formatter_error__with_message(self) -> None:
        """Test that FormatterError preserves error message.

        Raises:
            FormatterError: With a specific error message.
        """
        error_msg = "Failed to format transcription to SRT"
        with pytest.raises(FormatterError, match=error_msg):
            raise FormatterError(error_msg)

    def test_formatter_error__is_exception(self) -> None:
        """Test that FormatterError is a subclass of Exception."""
        assert issubclass(FormatterError, Exception)

    def test_formatter_error__empty_message(self) -> None:
        """Test that FormatterError can be raised with empty message.

        Raises:
            FormatterError: With no message.
        """
        with pytest.raises(FormatterError):
            raise FormatterError()


class TestExportError:
    """Test suite for ExportError exception."""

    def test_export_error__can_be_raised(self) -> None:
        """Test that ExportError can be raised.

        Raises:
            ExportError: To verify it can be caught.
        """
        with pytest.raises(ExportError):
            raise ExportError("Export failed")

    def test_export_error__with_message(self) -> None:
        """Test that ExportError preserves error message.

        Raises:
            ExportError: With a specific error message.
        """
        error_msg = "Failed to export transcription to file"
        with pytest.raises(ExportError, match=error_msg):
            raise ExportError(error_msg)

    def test_export_error__is_exception(self) -> None:
        """Test that ExportError is a subclass of Exception."""
        assert issubclass(ExportError, Exception)

    def test_export_error__empty_message(self) -> None:
        """Test that ExportError can be raised with empty message.

        Raises:
            ExportError: With no message.
        """
        with pytest.raises(ExportError):
            raise ExportError()


class TestExceptionInheritance:
    """Test suite for verifying exception inheritance relationships."""

    def test_all_errors__are_exceptions(self) -> None:
        """Test that all custom errors inherit from Exception."""
        custom_errors = [
            TranscriptionError,
            DeviceNotFoundError,
            FormatterError,
            ExportError,
        ]
        for error_class in custom_errors:
            assert issubclass(error_class, Exception)

    def test_errors__can_be_caught_as_exception(self) -> None:
        """Test that custom errors can be caught as base Exception.

        Raises:
            TranscriptionError: To verify it can be caught as Exception.
        """
        try:
            raise TranscriptionError("Test")
        except Exception as e:
            assert isinstance(e, TranscriptionError)
