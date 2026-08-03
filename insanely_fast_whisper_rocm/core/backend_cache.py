"""Shared backend/pipeline caching with reference counting.

This module provides a process-wide cache for ASR backends and pipelines so
repeated requests can reuse an in-memory model instead of reloading it each
time. It exposes acquire/release helpers that manage a reference count.

By default, entries are kept warm when their refcount drops to zero to maximize
reuse. Set the environment variable ``IFW_EAGER_MODEL_RELEASE=1`` to eagerly
close and remove cache entries when their refcount hits zero.

Alternatively, set ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS`` to a positive number
to release the backend after that many seconds of idleness. ``0`` releases
immediately. Unset, blank, malformed, negative, or non-finite values disable
the timeout (the model stays warm indefinitely). Eager release takes precedence
over the timeout and always releases immediately.
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
from insanely_fast_whisper_rocm.utils import constants

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
    _release_timer: threading.Timer | None = None
    _release_generation: int = 0


# Global cache keyed by an immutable config tuple
_CACHE: dict[tuple[Hashable, ...], _CacheEntry] = {}
_LOCK = threading.RLock()
_EAGER_RELEASE = constants.EAGER_MODEL_RELEASE
_RELEASE_TIMEOUT = constants.MODEL_RELEASE_TIMEOUT_SECONDS


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


def _cancel_release_timer(entry: _CacheEntry) -> None:
    """Cancel a pending delayed-release timer on *entry* if one exists."""
    if entry._release_timer is not None:
        entry._release_timer.cancel()
        entry._release_timer = None


def _schedule_release(
    key: tuple[Hashable, ...],
    entry: _CacheEntry,
    timeout: float,
) -> None:
    """Schedule a delayed release for a zero-refcount cache entry.

    Bumps the entry's release generation so any previously scheduled timer
    is invalidated, then starts a daemon timer.

    Args:
        key: The cache key identifying the entry.
        entry: The cache entry to release after the timeout.
        timeout: Delay in seconds before the backend is closed.
    """
    _cancel_release_timer(entry)
    entry._release_generation += 1
    generation = entry._release_generation
    timer = threading.Timer(timeout, _timed_release, args=(key, generation))
    timer.daemon = True
    entry._release_timer = timer
    timer.start()


def _timed_release(key: tuple[Hashable, ...], generation: int) -> None:
    """Timer callback: close and remove a cached backend after idle timeout.

    Validates the entry's identity (generation) and refcount under the lock,
    detaches it from the cache, then closes the backend outside the lock to
    avoid blocking other cache operations.

    Args:
        key: The cache key identifying the entry.
        generation: The release generation captured when the timer was scheduled.
    """
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return
        if entry._release_generation != generation:
            return
        if entry.ref_count != 0:
            return
        _CACHE.pop(key, None)
        entry._release_timer = None
        backend = entry.backend
    try:
        backend.close()
    except Exception as e:  # pragma: no cover - defensive cleanup
        logger.warning("Failed to close backend during timed release: %s", e)


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
        else:
            # Cancel any pending delayed release and invalidate stale timers.
            _cancel_release_timer(entry)
            entry._release_generation += 1
        entry.ref_count += 1
        return entry.pipeline, key


def release_pipeline(key: tuple[Hashable, ...]) -> None:
    """Release a previously acquired pipeline.

    Decrements the reference count for the cache entry. If it reaches zero,
    the backend is closed and removed according to the release policy:

    - ``IFW_EAGER_MODEL_RELEASE=1``: close immediately (overrides timeout).
    - ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS`` unset/invalid: keep warm.
    - ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS=0``: close immediately.
    - ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS=N`` (positive): close after *N*
      seconds of idleness via a cancellable background timer.

    Args:
        key: The cache key returned by ``acquire_pipeline``.
    """
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)
        if entry.ref_count > 0:
            return

        # refcount is zero — decide release policy.
        if _EAGER_RELEASE:
            _cancel_release_timer(entry)
            try:
                entry.backend.close()
            finally:
                _CACHE.pop(key, None)
            return

        if _RELEASE_TIMEOUT is None:
            # No timeout configured — keep the model warm.
            return

        if _RELEASE_TIMEOUT == 0:
            _cancel_release_timer(entry)
            try:
                entry.backend.close()
            finally:
                _CACHE.pop(key, None)
            return

        # Positive timeout — schedule a delayed, cancellable release.
        _schedule_release(key, entry, _RELEASE_TIMEOUT)


def invalidate_gpu_cache() -> None:
    """Close and remove all GPU-based pipelines from the cache.

    This is useful for OOM recovery when falling back to CPU, to ensure
    GPU memory is freed immediately.
    """
    with _LOCK:
        keys_to_remove = []
        for key, entry in _CACHE.items():
            # The key's second element is the device string
            device = key[1]
            if isinstance(device, str) and device != "cpu":
                _cancel_release_timer(entry)
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
    """Clear the entire cache.

    Args:
        force_close: If True, ``close()`` is called on all backends before
            removing entries.
    """
    with _LOCK:
        if force_close:
            for entry in _CACHE.values():
                _cancel_release_timer(entry)
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
