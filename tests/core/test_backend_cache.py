"""Comprehensive test suite for backend cache module.

This module tests the backend cache implementation including reference counting,
thread safety, and resource cleanup to prevent GPU memory leaks.
"""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, Mock, patch

from insanely_fast_whisper_rocm.core import backend_cache
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.backend_cache import (
    acquire_pipeline,
    borrow_pipeline,
    clear_cache,
    release_pipeline,
)


class TestBackendCache:
    """Test the backend cache reference counting and lifecycle."""

    def setup_method(self) -> None:
        """Clear the cache before each test."""
        clear_cache(force_close=True)

    def teardown_method(self) -> None:
        """Clean up cache after each test."""
        clear_cache(force_close=True)

    def test_acquire_pipeline_creates_entry(self) -> None:
        """Verify that acquire_pipeline creates a cache entry."""
        # Create a minimal config
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        # Mock backend and pipeline creation
        with patch(
            "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
        ) as mock_backend_class:
            with patch(
                "insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"
            ) as mock_pipeline_class:
                mock_backend = MagicMock()
                mock_pipeline = MagicMock()
                mock_backend_class.return_value = mock_backend
                mock_pipeline_class.return_value = mock_pipeline

                # Acquire a pipeline
                pipeline, key = acquire_pipeline(cfg)

                # Verify the cache has an entry
                assert key in backend_cache._CACHE
                assert backend_cache._CACHE[key].backend == mock_backend
                assert backend_cache._CACHE[key].pipeline == mock_pipeline
                assert backend_cache._CACHE[key].ref_count == 1

    def test_acquire_pipeline_increments_refcount(self) -> None:
        """Verify that acquiring the same pipeline increments ref_count."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                # Acquire twice
                pipeline1, key1 = acquire_pipeline(cfg)
                pipeline2, key2 = acquire_pipeline(cfg)

                # Keys should be the same
                assert key1 == key2

                # Ref count should be 2
                assert backend_cache._CACHE[key1].ref_count == 2

                # Both should return the same pipeline instance
                assert pipeline1 is pipeline2

    def test_release_pipeline_decrements_refcount(self) -> None:
        """Verify that release_pipeline decrements ref_count."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                # Acquire twice
                pipeline1, key1 = acquire_pipeline(cfg)
                pipeline2, key2 = acquire_pipeline(cfg)

                assert backend_cache._CACHE[key1].ref_count == 2

                # Release once
                release_pipeline(key1)

                # Ref count should be 1
                assert backend_cache._CACHE[key1].ref_count == 1

                # Release again
                release_pipeline(key1)

                # Ref count should be 0 (but entry should still exist in warm cache mode)
                assert backend_cache._CACHE[key1].ref_count == 0

    def test_eager_release_mode_closes_backend(self) -> None:
        """Verify that IFW_EAGER_MODEL_RELEASE=1 closes backend when ref_count hits 0."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        # Enable eager release mode
        with patch.dict(os.environ, {"IFW_EAGER_MODEL_RELEASE": "1"}):
            # Re-import to pick up the environment variable
            import importlib

            from insanely_fast_whisper_rocm import core

            importlib.reload(core.backend_cache)
            from insanely_fast_whisper_rocm.core.backend_cache import (
                acquire_pipeline as acquire_eager,
            )
            from insanely_fast_whisper_rocm.core.backend_cache import (
                release_pipeline as release_eager,
            )

            with patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mock_backend_class:
                with patch(
                    "insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"
                ):
                    mock_backend = MagicMock()
                    mock_backend.close = Mock()
                    mock_backend_class.return_value = mock_backend

                    # Acquire and release
                    pipeline, key = acquire_eager(cfg)
                    release_eager(key)

                    # Backend should have been closed
                    mock_backend.close.assert_called_once()

                    # Entry should be removed from cache
                    assert key not in backend_cache._CACHE

            # Reload back to normal mode
            importlib.reload(core.backend_cache)

    def test_warm_cache_mode_keeps_backend(self) -> None:
        """Verify that default warm cache behavior keeps backend alive at ref_count=0."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        # Ensure eager release is off for this test
        with patch.dict(os.environ, {"IFW_EAGER_MODEL_RELEASE": "0"}):
            # Re-import to pick up the environment variable
            import importlib

            from insanely_fast_whisper_rocm import core

            importlib.reload(core.backend_cache)
            from insanely_fast_whisper_rocm.core.backend_cache import (
                acquire_pipeline as acquire_warm,
            )
            from insanely_fast_whisper_rocm.core.backend_cache import (
                release_pipeline as release_warm,
            )

            with patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mock_backend_class:
                with patch(
                    "insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"
                ):
                    mock_backend = MagicMock()
                    mock_backend.close = Mock()
                    mock_backend_class.return_value = mock_backend

                    # Acquire and release
                    pipeline, key = acquire_warm(cfg)
                    release_warm(key)

                    # Backend should NOT have been closed (warm cache mode)
                    mock_backend.close.assert_not_called()

                    # Entry should still exist in cache
                    assert key in backend_cache._CACHE
                    assert backend_cache._CACHE[key].ref_count == 0

            # Reload back to normal
            importlib.reload(core.backend_cache)

    def test_borrow_pipeline_context_manager(self) -> None:
        """Verify that borrow_pipeline context manager properly acquires and releases."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch(
                "insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = MagicMock()
                mock_pipeline_class.return_value = mock_pipeline

                # Use the context manager
                with borrow_pipeline(cfg) as pipeline:
                    # Should have acquired the pipeline
                    assert pipeline == mock_pipeline

                    # Find the key for this config
                    keys = list(backend_cache._CACHE.keys())
                    assert len(keys) == 1
                    key = keys[0]

                    # Ref count should be 1
                    assert backend_cache._CACHE[key].ref_count == 1

                # After exiting, ref count should be 0 (if entry still exists)
                # The entry may be removed by eager release mode
                if key in backend_cache._CACHE:
                    assert backend_cache._CACHE[key].ref_count == 0

    def test_borrow_pipeline_releases_on_exception(self) -> None:
        """Verify that borrow_pipeline releases even if an exception occurs.

        Raises:
            RuntimeError: Test exception to verify cleanup happens.
        """
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                # Use the context manager with an exception
                key = None
                try:
                    with borrow_pipeline(cfg):
                        keys = list(backend_cache._CACHE.keys())
                        key = keys[0]

                        # Ref count should be 1
                        assert backend_cache._CACHE[key].ref_count == 1

                        # Raise an exception
                        raise RuntimeError("Test exception")
                except RuntimeError:
                    pass

                # After exception, ref count should still be 0 (cleanup happened)
                # The entry may be removed by eager release mode
                if key and key in backend_cache._CACHE:
                    assert backend_cache._CACHE[key].ref_count == 0

    def test_clear_cache_with_force_close(self) -> None:
        """Verify that clear_cache(force_close=True) closes all backends."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch(
            "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
        ) as mock_backend_class:
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                mock_backend = MagicMock()
                mock_backend.close = Mock()
                mock_backend_class.return_value = mock_backend

                # Acquire a pipeline
                pipeline, key = acquire_pipeline(cfg)

                # Clear the cache with force_close
                clear_cache(force_close=True)

                # Backend should have been closed
                mock_backend.close.assert_called_once()

                # Cache should be empty
                assert len(backend_cache._CACHE) == 0

    def test_clear_cache_logs_exceptions(self) -> None:
        """Verify that exceptions during cache clear are logged."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch(
            "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
        ) as mock_backend_class:
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                mock_backend = MagicMock()
                # Make close() raise an exception
                mock_backend.close = Mock(side_effect=RuntimeError("Close failed"))
                mock_backend_class.return_value = mock_backend

                # Acquire a pipeline
                pipeline, key = acquire_pipeline(cfg)

                # Clear the cache with force_close (should not crash)
                with patch(
                    "insanely_fast_whisper_rocm.core.backend_cache.logger"
                ) as mock_logger:
                    clear_cache(force_close=True)

                    # Logger should have been called with a warning
                    assert mock_logger.warning.called
                    warning_call = mock_logger.warning.call_args
                    assert "Failed to close backend" in str(warning_call)

                # Cache should still be cleared despite the exception
                assert len(backend_cache._CACHE) == 0

    def test_concurrent_access_thread_safety(self) -> None:
        """Verify that concurrent access to the cache is thread-safe."""
        cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                # Acquire and release from multiple threads
                errors = []

                def worker() -> None:
                    try:
                        for _ in range(10):
                            pipeline, key = acquire_pipeline(cfg)
                            release_pipeline(key)
                    except Exception as e:
                        errors.append(e)

                threads = [threading.Thread(target=worker) for _ in range(5)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # No errors should have occurred
                assert len(errors) == 0

                # Final ref count should be 0 (if entry exists)
                # The entry may be removed by eager release mode
                keys = list(backend_cache._CACHE.keys())
                if len(keys) > 0:
                    assert len(keys) == 1
                    assert backend_cache._CACHE[keys[0]].ref_count == 0

    def test_cache_key_generation_stability(self) -> None:
        """Verify that the same config generates the same cache key."""
        cfg1 = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        cfg2 = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )

        with patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"):
            with patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"):
                # Acquire with both configs
                pipeline1, key1 = acquire_pipeline(cfg1)
                pipeline2, key2 = acquire_pipeline(cfg2)

                # Keys should be identical
                assert key1 == key2

                # Should return the same pipeline instance
                assert pipeline1 is pipeline2

                # Only one entry in cache
                assert len(backend_cache._CACHE) == 1
