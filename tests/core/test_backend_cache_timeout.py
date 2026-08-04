"""Tests for configurable idle-timeout model release in backend_cache.

Covers the ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS`` feature: immediate release
(timeout=0), delayed release via a cancellable timer, reacquire race,
generation-guarded stale-timer suppression, eager-release precedence, and
interaction with forceful cache clearing / GPU invalidation.

The tests use a fake ``threading.Timer`` so the delayed-release callback runs
synchronously and deterministically without real wall-clock waits.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from importlib import reload
from unittest.mock import MagicMock, Mock, patch

import pytest

from insanely_fast_whisper_rocm.core import backend_cache
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.backend_cache import (
    acquire_pipeline,
    borrow_pipeline,
    clear_cache,
    invalidate_gpu_cache,
    release_pipeline,
)


def _cfg() -> HuggingFaceBackendConfig:
    """Return a minimal backend config for cache tests."""
    return HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=1,
        chunk_length=30,
        progress_group_size=5,
    )


class FakeTimer:
    """Deterministic replacement for ``threading.Timer``.

    The callback is stored but *not* executed until :meth:`fire` is called,
    so tests control exactly when (or whether) the delayed release runs.
    """

    instances: list[FakeTimer] = []

    def __init__(
        self,
        interval: float,
        function: object,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> None:
        """Store the timer callback without executing it.

        Args:
            interval: Delay in seconds (unused, kept for API compatibility).
            function: The callable to invoke on :meth:`fire`.
            args: Positional arguments passed to *function*.
            kwargs: Keyword arguments passed to *function*.
        """
        self.interval = interval
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.cancelled = False
        self.started = False
        FakeTimer.instances.append(self)

    def start(self) -> None:
        """Mark the timer as started (no thread is created)."""
        self.started = True

    def cancel(self) -> None:
        """Mark the timer as cancelled."""
        self.cancelled = True

    def fire(self) -> None:
        """Execute the timer callback.

        Fires even if the timer was cancelled, simulating the race where
        ``cancel()`` lost to an already-running scheduler.  The production
        generation guard is the real defense; this just lets tests exercise
        that defense-in-depth path.
        """
        self.function(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear cache and fake-timer log before each test.

    Yields:
        None: Control back to the test; after the test, cache is re-cleared.
    """
    clear_cache(force_close=True)
    FakeTimer.instances.clear()
    yield
    clear_cache(force_close=True)
    FakeTimer.instances.clear()


@pytest.fixture
def fake_timer(monkeypatch: pytest.MonkeyPatch) -> type:
    """Patch ``threading.Timer`` in backend_cache with FakeTimer.

    Yields:
        The FakeTimer class so tests can inspect created instances.
    """
    monkeypatch.setattr(backend_cache.threading, "Timer", FakeTimer)
    yield FakeTimer
    monkeypatch.setattr(backend_cache.threading, "Timer", threading.Timer)


@pytest.fixture
def release_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[bool, float | None], None]:
    """Configure release globals and restore them after the test.

    Returns:
        A function that applies eager-release and timeout values.
    """

    def configure(eager_release: bool, release_timeout: float | None) -> None:
        """Set the release policy for the current test."""
        monkeypatch.setattr(backend_cache, "_EAGER_RELEASE", eager_release)
        monkeypatch.setattr(backend_cache, "_RELEASE_TIMEOUT", release_timeout)

    return configure


class TestReleasePolicyIsolation:
    """Test release-policy overrides used by timeout tests."""

    def test_release_policy__restores_globals_with_monkeypatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        release_policy: Callable[[bool, float | None], None],
    ) -> None:
        """Configured release globals are registered for automatic restoration."""
        original_eager = backend_cache._EAGER_RELEASE
        original_timeout = backend_cache._RELEASE_TIMEOUT

        release_policy(True, 30.0)

        assert backend_cache._EAGER_RELEASE is True
        assert backend_cache._RELEASE_TIMEOUT == 30.0
        monkeypatch.undo()
        assert backend_cache._EAGER_RELEASE is original_eager
        assert backend_cache._RELEASE_TIMEOUT == original_timeout


# ---------------------------------------------------------------------------
# Constants parsing
# ---------------------------------------------------------------------------


class TestTimeoutParsing:
    """Test ``IFW_MODEL_RELEASE_TIMEOUT_SECONDS`` parsing in constants.py."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("abc", None),
            ("-1", None),
            ("-0.1", None),
            ("inf", None),
            ("-inf", None),
            ("nan", None),
            ("0", 0.0),
            ("0.0", 0.0),
            ("300", 300.0),
            ("1.5", 1.5),
        ],
    )
    def test_parsing(
        self, raw: str | None, expected: float | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that each raw value maps to the expected parsed result."""
        import insanely_fast_whisper_rocm.utils.constants as const_mod

        if raw is None:
            monkeypatch.delenv("IFW_MODEL_RELEASE_TIMEOUT_SECONDS", raising=False)
        else:
            monkeypatch.setenv("IFW_MODEL_RELEASE_TIMEOUT_SECONDS", raw)

        # Reload to re-evaluate the constant.
        reload(const_mod)
        assert const_mod.MODEL_RELEASE_TIMEOUT_SECONDS == expected

        # Restore module state for subsequent tests.
        monkeypatch.delenv("IFW_MODEL_RELEASE_TIMEOUT_SECONDS", raising=False)
        reload(const_mod)


# ---------------------------------------------------------------------------
# Backend cache timeout behavior
# ---------------------------------------------------------------------------


class TestTimeoutRelease:
    """Test idle-timeout release behavior in the backend cache."""

    def test_timeout_zero_releases_immediately(
        self, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """Timeout=0 closes the backend immediately on release."""
        release_policy(False, 0.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            mock_backend.close.assert_called_once()
            assert key not in backend_cache._CACHE

    def test_positive_timeout_schedules_timer(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """A positive timeout schedules a daemon timer but does not close yet."""
        release_policy(False, 300.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            # Entry still in cache (warm, waiting for timer).
            assert key in backend_cache._CACHE
            assert backend_cache._CACHE[key].ref_count == 0
            mock_backend.close.assert_not_called()

            # A timer was scheduled with the correct interval.
            assert len(fake_timer.instances) == 1
            timer = fake_timer.instances[0]
            assert timer.interval == 300.0
            assert timer.started
            assert not timer.cancelled

    def test_timer_fires_closes_and_removes(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """When the timer fires, the backend is closed and entry removed."""
        release_policy(False, 60.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            assert len(fake_timer.instances) == 1
            fake_timer.instances[0].fire()

            mock_backend.close.assert_called_once()
            assert key not in backend_cache._CACHE

    def test_reacquire_cancels_timer(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """Reacquiring a pipeline cancels the pending release timer."""
        release_policy(False, 120.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            timer = fake_timer.instances[0]
            assert not timer.cancelled

            # Reacquire — should cancel the timer.
            _pipeline2, _key2 = acquire_pipeline(_cfg())
            assert timer.cancelled
            assert backend_cache._CACHE[key].ref_count == 1
            mock_backend.close.assert_not_called()

    def test_stale_timer_is_noop(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """A timer from a previous generation does not close the backend."""
        release_policy(False, 10.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            stale_timer = fake_timer.instances[0]
            stale_generation = backend_cache._CACHE[key]._release_generation

            # Reacquire (bumps generation, cancels stale timer).
            _pipeline2, _key2 = acquire_pipeline(_cfg())
            current_gen = backend_cache._CACHE[key]._release_generation
            assert current_gen != stale_generation

            # Simulate the stale timer firing despite cancellation.
            stale_timer.fire()

            # Backend must not have been closed by the stale timer.
            mock_backend.close.assert_not_called()
            assert key in backend_cache._CACHE
            assert backend_cache._CACHE[key].ref_count == 1

    def test_stale_timer_does_not_close_recreated_entry(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """A stale timer cannot close a newer idle entry with the same key."""
        release_policy(False, 10.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            first_backend = MagicMock()
            first_backend.close = Mock()
            recreated_backend = MagicMock()
            recreated_backend.close = Mock()
            mb.side_effect = [first_backend, recreated_backend]

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)
            stale_timer = fake_timer.instances[0]

            clear_cache(force_close=True)
            _pipeline, recreated_key = acquire_pipeline(_cfg())
            release_pipeline(recreated_key)

            assert recreated_key == key
            assert len(fake_timer.instances) == 2
            assert backend_cache._CACHE[key]._release_generation != stale_timer.args[1]

            stale_timer.fire()

            recreated_backend.close.assert_not_called()
            assert key in backend_cache._CACHE

    def test_eager_overrides_timeout(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """Eager release takes precedence over a positive timeout."""
        release_policy(True, 300.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            mock_backend.close.assert_called_once()
            assert key not in backend_cache._CACHE
            # No timer should have been scheduled.
            assert len(fake_timer.instances) == 0

    def test_no_timeout_keeps_warm(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """When timeout is None, release keeps the model warm (no timer)."""
        release_policy(False, None)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            mock_backend.close.assert_not_called()
            assert key in backend_cache._CACHE
            assert backend_cache._CACHE[key].ref_count == 0
            assert len(fake_timer.instances) == 0

    def test_borrow_pipeline_with_timeout(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """borrow_pipeline schedules a timer on exit when timeout is positive."""
        release_policy(False, 45.0)
        with (
            patch("insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"),
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            with borrow_pipeline(_cfg()):
                pass

            assert len(fake_timer.instances) == 1
            assert not fake_timer.instances[0].cancelled


# ---------------------------------------------------------------------------
# Forceful clearing cancels pending timers
# ---------------------------------------------------------------------------


class TestForcefulClearCancelsTimers:
    """Verify that clear_cache and invalidate_gpu_cache cancel pending timers."""

    def test_clear_cache_cancels_timer(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """clear_cache(force_close=True) cancels pending release timers."""
        release_policy(False, 90.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            timer = fake_timer.instances[0]
            assert not timer.cancelled

            clear_cache(force_close=True)

            assert timer.cancelled
            assert len(backend_cache._CACHE) == 0

    def test_invalidate_gpu_cache_cancels_timer(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """invalidate_gpu_cache cancels pending release timers for GPU entries."""
        release_policy(False, 90.0)
        gpu_cfg = HuggingFaceBackendConfig(
            model_name="openai/whisper-tiny",
            device="cuda:0",
            dtype="float16",
            batch_size=1,
            chunk_length=30,
            progress_group_size=5,
        )
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(gpu_cfg)
            release_pipeline(key)

            timer = fake_timer.instances[0]
            assert not timer.cancelled

            invalidate_gpu_cache()

            assert timer.cancelled
            assert key not in backend_cache._CACHE


# ---------------------------------------------------------------------------
# Timer fires but entry was already removed — no error
# ---------------------------------------------------------------------------


class TestTimedReleaseEdgeCases:
    """Edge cases for the timed-release callback."""

    def test_timed_release_missing_entry_is_noop(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """If the entry was already removed, the timer is a no-op."""
        release_policy(False, 10.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            timer = fake_timer.instances[0]

            # Remove the entry externally (simulates concurrent clear).
            with backend_cache._LOCK:
                backend_cache._CACHE.pop(key, None)

            # Firing the timer must not raise.
            timer.fire()
            mock_backend.close.assert_not_called()

    def test_timed_release_nonzero_refcount_is_noop(
        self, fake_timer: type, release_policy: Callable[[bool, float | None], None]
    ) -> None:
        """If refcount is non-zero when the timer fires, it is a no-op."""
        release_policy(False, 10.0)
        with (
            patch(
                "insanely_fast_whisper_rocm.core.backend_cache.HuggingFaceBackend"
            ) as mb,
            patch("insanely_fast_whisper_rocm.core.backend_cache.WhisperPipeline"),
        ):
            mock_backend = MagicMock()
            mock_backend.close = Mock()
            mb.return_value = mock_backend

            _pipeline, key = acquire_pipeline(_cfg())
            release_pipeline(key)

            timer = fake_timer.instances[0]

            # Reacquire (refcount goes to 1, generation bumped, old timer cancelled).
            _pipeline2, _key2 = acquire_pipeline(_cfg())
            assert timer.cancelled

            # Firing the stale (cancelled) timer should be a no-op.
            timer.fire()
            mock_backend.close.assert_not_called()
            assert key in backend_cache._CACHE
            assert backend_cache._CACHE[key].ref_count == 1
