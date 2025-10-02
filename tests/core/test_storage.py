"""Tests for insanely_fast_whisper_api.core.storage module.

This module contains tests for storage backend abstractions and implementations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_api.core.storage import (
    BaseStorage,
    JsonStorage,
    StorageFactory,
)


class TestJsonStorage:
    """Test suite for JsonStorage class."""

    def test_json_storage__save_success(self) -> None:
        """Test successful save operation."""
        storage = JsonStorage()
        data = {"text": "Test transcription", "segments": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "output" / "test.json"
            result = storage.save(data, dest_path, "transcribe")

            assert result is not None
            assert Path(result).exists()

            # Verify content
            with open(result, encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data == data

    def test_json_storage__creates_parent_directory(self) -> None:
        """Test that save creates parent directory if it doesn't exist."""
        storage = JsonStorage()
        data = {"text": "Test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "nested" / "dir" / "test.json"
            assert not dest_path.parent.exists()

            result = storage.save(data, dest_path, "transcribe")

            assert result is not None
            assert dest_path.parent.exists()
            assert Path(result).exists()

    def test_json_storage__adds_json_extension(self) -> None:
        """Test that save adds .json extension if missing."""
        storage = JsonStorage()
        data = {"text": "Test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test"  # No extension
            result = storage.save(data, dest_path, "transcribe")

            assert result is not None
            assert result.endswith(".json")

    def test_json_storage__preserves_json_extension(self) -> None:
        """Test that save preserves existing .json extension."""
        storage = JsonStorage()
        data = {"text": "Test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test.json"
            result = storage.save(data, dest_path, "transcribe")

            assert result is not None
            assert result.endswith(".json")
            assert result.count(".json") == 1  # Not duplicated

    def test_json_storage__save_unicode_content(self) -> None:
        """Test saving unicode content."""
        storage = JsonStorage()
        data = {"text": "Test with émojis 🎉 and spëcial çhars"}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test.json"
            result = storage.save(data, dest_path, "transcribe")

            assert result is not None

            with open(result, encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data["text"] == data["text"]

    def test_json_storage__save_complex_data(self) -> None:
        """Test saving complex nested data structures."""
        storage = JsonStorage()
        data = {
            "text": "Complex test",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "First"},
                {"start": 1.5, "end": 3.0, "text": "Second"},
            ],
            "metadata": {"model": "whisper-tiny", "language": "en"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test.json"
            result = storage.save(data, dest_path, "transcribe")

            assert result is not None

            with open(result, encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data == data

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_json_storage__save_handles_os_error(
        self, mock_file_open: MagicMock
    ) -> None:
        """Test that save handles OSError gracefully."""
        storage = JsonStorage()
        data = {"text": "Test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "test.json"
            result = storage.save(data, dest_path, "transcribe")

            assert result is None

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", side_effect=OSError("Disk full"))
    def test_json_storage__save_logs_error(
        self, mock_file_open: MagicMock, mock_mkdir: MagicMock
    ) -> None:
        """Test that save logs errors appropriately."""
        storage = JsonStorage()
        data = {"text": "Test"}
        dest_path = Path("/tmp/test.json")

        with patch("insanely_fast_whisper_api.core.storage.logger") as mock_logger:
            result = storage.save(data, dest_path, "transcribe")

            assert result is None
            mock_logger.error.assert_called_once()
            # Verify error was logged with exc_info
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs.get("exc_info") is True


class TestStorageFactory:
    """Test suite for StorageFactory class."""

    def test_storage_factory__create_json_storage(self) -> None:
        """Test creating JSON storage through factory."""
        storage = StorageFactory.create("json")
        assert isinstance(storage, JsonStorage)

    def test_storage_factory__create_json_storage_default(self) -> None:
        """Test creating JSON storage with default parameter."""
        storage = StorageFactory.create()
        assert isinstance(storage, JsonStorage)

    def test_storage_factory__raises_for_unsupported_kind(self) -> None:
        """Test that factory raises ValueError for unsupported storage kind."""
        with pytest.raises(ValueError, match="Unsupported storage kind: invalid"):
            StorageFactory.create("invalid")

    def test_storage_factory__raises_for_future_types(self) -> None:
        """Test that factory raises ValueError for not-yet-implemented types."""
        with pytest.raises(ValueError, match="Unsupported storage kind"):
            StorageFactory.create("sqlite")


class TestBaseStorage:
    """Test suite for BaseStorage abstract class."""

    def test_base_storage__is_abstract(self) -> None:
        """Test that BaseStorage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStorage()  # type: ignore[abstract]

    def test_base_storage__requires_save_implementation(self) -> None:
        """Test that subclasses must implement save method."""

        class IncompleteStorage(BaseStorage):
            """Test subclass without save implementation."""

            pass

        with pytest.raises(TypeError):
            IncompleteStorage()  # type: ignore[abstract]

    def test_base_storage__concrete_subclass_works(self) -> None:
        """Test that properly implemented subclass can be instantiated."""

        class ConcreteStorage(BaseStorage):
            """Test subclass with save implementation."""

            def save(
                self, data: dict[str, Any], destination_path: Path, task: str
            ) -> str | None:
                """Concrete implementation of save.

                Returns:
                    Path as string.
                """
                return str(destination_path)

        # Should not raise
        storage = ConcreteStorage()
        assert isinstance(storage, BaseStorage)
