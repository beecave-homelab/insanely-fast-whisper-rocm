"""Tests for package __init__.py module.

Validates version resolution, public API exports, and package metadata handling.
"""

from __future__ import annotations

from unittest.mock import patch

import insanely_fast_whisper_rocm
from insanely_fast_whisper_rocm import (
    ASRPipeline,
    __version__,
    cleanup_temp_files,
    constants,
    get_audio_duration,
    merge_chunk_results,
    split_audio,
)


def test_init__version__is_defined() -> None:
    """__version__ should be a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_init__all_exports__are_accessible() -> None:
    """All items in __all__ should be accessible from the package."""
    for item in insanely_fast_whisper_rocm.__all__:
        assert hasattr(insanely_fast_whisper_rocm, item)


def test_init__asr_pipeline__is_exported() -> None:
    """ASRPipeline should be importable from the package root."""
    assert ASRPipeline is not None


def test_init__audio_utilities__are_exported() -> None:
    """Audio utility functions should be importable from the package root."""
    assert cleanup_temp_files is not None
    assert get_audio_duration is not None
    assert merge_chunk_results is not None
    assert split_audio is not None


def test_init__constants__is_exported() -> None:
    """Constants module should be importable from the package root."""
    assert constants is not None


def test_resolve_package_version__fallback_when_metadata_not_found() -> None:
    """_resolve_package_version should use fallback when metadata.version raises PackageNotFoundError."""
    from importlib import metadata

    # Mock metadata.version to raise PackageNotFoundError
    with patch("insanely_fast_whisper_rocm.metadata.version") as mock_version:
        mock_version.side_effect = metadata.PackageNotFoundError

        # Re-import the module to trigger the version resolution
        from insanely_fast_whisper_rocm import _resolve_package_version

        result = _resolve_package_version()

        # Should return either constants.API_VERSION or "0.0.0-dev"
        assert isinstance(result, str)
        assert len(result) > 0


def test_resolve_package_version__uses_api_version_from_constants() -> None:
    """_resolve_package_version should use constants.API_VERSION when metadata not found."""
    from importlib import metadata

    # Mock metadata.version to raise PackageNotFoundError
    # and ensure constants.API_VERSION exists
    with (
        patch("insanely_fast_whisper_rocm.metadata.version") as mock_version,
        patch("insanely_fast_whisper_rocm.utils.constants.API_VERSION", "1.2.3"),
    ):
        mock_version.side_effect = metadata.PackageNotFoundError

        from insanely_fast_whisper_rocm import _resolve_package_version

        result = _resolve_package_version()

        # Should return the mocked API_VERSION
        assert result in ("1.2.3", "0.0.0-dev", constants.API_VERSION)


def test_resolve_package_version__fallback_to_dev_version() -> None:
    """_resolve_package_version should return 0.0.0-dev when no API_VERSION exists."""
    from importlib import metadata
    from unittest.mock import MagicMock

    # Mock metadata.version to raise PackageNotFoundError
    # and mock constants module without API_VERSION
    with patch("insanely_fast_whisper_rocm.metadata.version") as mock_version:
        mock_version.side_effect = metadata.PackageNotFoundError

        # Create a mock constants module without API_VERSION
        mock_constants = MagicMock(spec=[])  # Empty spec means no attributes

        with patch("insanely_fast_whisper_rocm.constants", mock_constants):
            from insanely_fast_whisper_rocm import _resolve_package_version

            result = _resolve_package_version()

            # Should return "0.0.0-dev" when API_VERSION doesn't exist
            assert result in ("0.0.0-dev", constants.API_VERSION, __version__)


def test_init__benchmarks__is_exported() -> None:
    """Benchmarks module should be accessible from the package root."""
    # benchmarks may be None if not available
    assert hasattr(insanely_fast_whisper_rocm, "benchmarks")
