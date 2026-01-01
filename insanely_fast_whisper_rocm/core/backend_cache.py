"""Shared backend/pipeline caching with reference counting.

This module provides a process-wide cache for ASR backends and pipelines so
repeated requests can reuse an in-memory model instead of reloading it each
time. It exposes acquire/release helpers that manage a reference count.

By default, entries are kept warm when their refcount drops to zero to maximize
reuse. Set the environment variable ``IFW_EAGER_MODEL_RELEASE=1`` to eagerly
close and remove cache entries when their refcount hits zero.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Hashable, Iterator
from dataclasses import dataclass

from insanely_fast_whisper_rocm.core.asr_backend import (
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
)
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Container for a cached backend/pipeline pair.

    Args:
        backend: The cached ASR backend instance.
        pipeline: A pipeline bound to the backend for end-to-end processing.
        ref_count: Number of active borrowers for this pipeline.
    """

    backend: HuggingFaceBackend
    pipeline: WhisperPipeline
    ref_count: int = 0


# Global cache keyed by an immutable config tuple
_CACHE: dict[tuple[Hashable, ...], _CacheEntry] = {}
_LOCK = threading.RLock()
_EAGER_RELEASE = os.getenv("IFW_EAGER_MODEL_RELEASE", "0") in ("1", "true", "True")


def _make_key(
    cfg: HuggingFaceBackendConfig,
    *,
    save_transcriptions: bool,
    output_dir: str,
) -> tuple[Hashable, ...]:
    """Create a stable key for the given backend configuration.

    Args:
        cfg: Backend configuration.
        save_transcriptions: Whether the pipeline persists JSON outputs.
        output_dir: Directory that stores transcript artefacts.

    Returns:
        A tuple suitable for use as a dict key.
    """
    return (
        cfg.model_name,
        cfg.device,
        cfg.dtype,
        int(cfg.batch_size),
        int(cfg.chunk_length),
        int(cfg.progress_group_size),
        bool(save_transcriptions),
        os.path.abspath(output_dir),
    )


def acquire_pipeline(
    cfg: HuggingFaceBackendConfig,
    *,
    save_transcriptions: bool = True,
    output_dir: str = "transcripts",
) -> tuple[WhisperPipeline, tuple[Hashable, ...]]:
    """Get a cached pipeline for the config, creating it if necessary.

    Increments the cache entry's reference count and returns the pipeline and
    its key. Call ``release_pipeline(key)`` when done.

    Args:
        cfg: Backend configuration.
        save_transcriptions: Whether the pipeline should persist JSON outputs.
        output_dir: Directory used by the pipeline when saving results.

    Returns:
        A (pipeline, key) tuple.
    """
    normalized_output_dir = os.path.abspath(output_dir)
    key = _make_key(
        cfg,
        save_transcriptions=save_transcriptions,
        output_dir=normalized_output_dir,
    )
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            backend = HuggingFaceBackend(config=cfg)
            pipeline = WhisperPipeline(
                asr_backend=backend,
                save_transcriptions=save_transcriptions,
                output_dir=normalized_output_dir,
            )
            entry = _CacheEntry(backend=backend, pipeline=pipeline, ref_count=0)
            _CACHE[key] = entry
        entry.ref_count += 1
        return entry.pipeline, key


def release_pipeline(key: tuple[Hashable, ...]) -> None:
    """Release a previously acquired pipeline.

    Decrements the reference count for the cache entry. If it reaches zero and
    eager release is enabled (``IFW_EAGER_MODEL_RELEASE=1``), the backend is
    closed and the entry is removed from the cache.

    Args:
        key: The cache key returned by ``acquire_pipeline``.
    """
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        if entry.ref_count == 0 and _EAGER_RELEASE:
            try:
                entry.backend.close()
            finally:
                _CACHE.pop(key, None)


def invalidate_gpu_cache() -> None:
    """
    Invalidate all GPU-backed cache entries by closing their backends and removing them from the cache.
    
    This frees GPU memory immediately and is intended for out-of-memory recovery when falling back to CPU.
    """
    with _LOCK:
        keys_to_remove = []
        for key, entry in _CACHE.items():
            # The key's second element is the device string
            device = key[1]
            if isinstance(device, str) and device != "cpu":
                try:
                    logger.info("Invalidating GPU cache entry for device: %s", device)
                    entry.backend.close()
                except Exception as e:
                    logger.warning(
                        "Failed to close GPU backend during invalidation: %s", e
                    )
                keys_to_remove.append(key)

        for key in keys_to_remove:
            _CACHE.pop(key, None)


def clear_cache(force_close: bool = False) -> None:
    """
    Clear all cached backend and pipeline entries.
    
    If `force_close` is True, attempt to call `close()` on each backend before removing entries; otherwise entries are removed without closing.
    
    Parameters:
        force_close (bool): When True, attempt to close every backend before clearing the cache.
    """
    with _LOCK:
        if force_close:
            for entry in _CACHE.values():
                try:
                    entry.backend.close()
                except Exception as e:  # pragma: no cover - defensive cleanup
                    # Log the exception with stack trace for debugging
                    logger.warning(
                        "Failed to close backend during cache clear: %s",
                        e,
                        exc_info=True,
                    )
        _CACHE.clear()


@contextlib.contextmanager
def borrow_pipeline(
    cfg: HuggingFaceBackendConfig,
    *,
    save_transcriptions: bool = True,
    output_dir: str = "transcripts",
) -> Iterator[WhisperPipeline]:
    """Context manager that acquires and releases a cached pipeline safely.

    Args:
        cfg: Backend configuration used for cache lookup.
        save_transcriptions: Whether the borrowed pipeline should persist results.
        output_dir: Destination directory for persisted transcripts.

    Yields:
        WhisperPipeline: The cached pipeline instance matching ``cfg``.
    """
    pipeline, key = acquire_pipeline(
        cfg,
        save_transcriptions=save_transcriptions,
        output_dir=output_dir,
    )
    try:
        yield pipeline
    finally:
        release_pipeline(key)