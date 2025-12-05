"""Tests for insanely_fast_whisper_rocm.core.cancellation module.

This module contains tests for cooperative cancellation utilities including
CancellationToken, CancellationScope, and cancellation_scope context manager.
"""

from __future__ import annotations

import pytest

from insanely_fast_whisper_rocm.core.cancellation import (
    CancellationScope,
    CancellationToken,
    cancellation_scope,
)
from insanely_fast_whisper_rocm.core.errors import TranscriptionCancelledError


class TestCancellationToken:
    """Test suite for CancellationToken class."""

    def test_init__starts_uncancelled(self) -> None:
        """Test that a new token starts in uncancelled state."""
        token = CancellationToken()

        assert token.cancelled is False

    def test_cancel__sets_cancelled_flag(self) -> None:
        """Test that cancel() sets the cancelled flag to True."""
        token = CancellationToken()

        token.cancel()

        assert token.cancelled is True

    def test_cancel__idempotent(self) -> None:
        """Test that calling cancel() multiple times is safe."""
        token = CancellationToken()

        token.cancel()
        token.cancel()

        assert token.cancelled is True

    def test_raise_if_cancelled__does_nothing_when_not_cancelled(self) -> None:
        """Test that raise_if_cancelled does nothing when token is not cancelled."""
        token = CancellationToken()

        # Should not raise
        token.raise_if_cancelled()

    def test_raise_if_cancelled__raises_when_cancelled(self) -> None:
        """Test that raise_if_cancelled raises when token is cancelled."""
        token = CancellationToken()
        token.cancel()

        with pytest.raises(
            TranscriptionCancelledError, match="Transcription cancelled by caller"
        ):
            token.raise_if_cancelled()


class TestCancellationScope:
    """Test suite for CancellationScope class."""

    def test_init__stores_token(self) -> None:
        """Test that CancellationScope stores the provided token."""
        token = CancellationToken()
        scope = CancellationScope(token)

        assert scope.token is token

    def test_cancel__calls_token_cancel(self) -> None:
        """Test that scope.cancel() calls token.cancel()."""
        token = CancellationToken()
        scope = CancellationScope(token)

        scope.cancel()

        assert token.cancelled is True

    def test_enter__returns_token(self) -> None:
        """Test that __enter__ returns the managed token."""
        token = CancellationToken()
        scope = CancellationScope(token)

        result = scope.__enter__()

        assert result is token

    def test_exit__no_exception__does_not_cancel(self) -> None:
        """Test that __exit__ does not cancel when no exception occurs."""
        token = CancellationToken()
        scope = CancellationScope(token)

        scope.__enter__()
        suppressed = scope.__exit__(None, None, None)

        assert token.cancelled is False
        assert suppressed is False

    def test_exit__cancellation_error__does_not_cancel_token(self) -> None:
        """Test that __exit__ does not cancel token for TranscriptionCancelledError."""
        token = CancellationToken()
        scope = CancellationScope(token)

        error = TranscriptionCancelledError("Test cancellation")
        scope.__enter__()
        suppressed = scope.__exit__(type(error), error, None)

        # Should not cancel the token since it's already a cancellation error
        assert token.cancelled is False
        # Should not suppress the exception
        assert suppressed is False

    def test_exit__other_exception__cancels_token(self) -> None:
        """Test that __exit__ cancels token for non-cancellation exceptions."""
        token = CancellationToken()
        scope = CancellationScope(token)

        error = ValueError("Test error")
        scope.__enter__()
        suppressed = scope.__exit__(type(error), error, None)

        # Should cancel the token to signal downstream tasks
        assert token.cancelled is True
        # Should not suppress the exception
        assert suppressed is False

    def test_context_manager__normal_flow(self) -> None:
        """Test using CancellationScope as a context manager."""
        token = CancellationToken()
        scope = CancellationScope(token)

        with scope as ctx_token:
            assert ctx_token is token
            assert token.cancelled is False

        assert token.cancelled is False

    def test_context_manager__with_exception(self) -> None:
        """Test CancellationScope context manager with exception.

        Raises:
            RuntimeError: Expected test exception.
        """
        token = CancellationToken()
        scope = CancellationScope(token)

        with pytest.raises(RuntimeError):
            with scope:
                raise RuntimeError("Test error")

        # Token should be cancelled due to exception
        assert token.cancelled is True


class TestCancellationScopeFunction:
    """Test suite for cancellation_scope function."""

    def test_with_new_token__creates_and_returns_token(self) -> None:
        """Test that cancellation_scope creates a new token when none provided."""
        with cancellation_scope() as token:
            assert isinstance(token, CancellationToken)
            assert token.cancelled is False

    def test_with_existing_token__reuses_token(self) -> None:
        """Test that cancellation_scope reuses the provided token."""
        existing_token = CancellationToken()

        with cancellation_scope(existing_token) as token:
            assert token is existing_token

    def test_normal_exit__does_not_cancel(self) -> None:
        """Test that normal exit does not cancel the token."""
        with cancellation_scope() as token:
            pass

        assert token.cancelled is False

    def test_exception_exit__cancels_token(self) -> None:
        """Test that exception exit cancels the token.

        Raises:
            ValueError: Expected test exception.
        """
        token = None
        with pytest.raises(ValueError):
            with cancellation_scope() as t:
                token = t
                raise ValueError("Test error")

        assert token is not None
        assert token.cancelled is True

    def test_cancellation_error__does_not_cancel(self) -> None:
        """Test that TranscriptionCancelledError does not cancel the token.

        Raises:
            TranscriptionCancelledError: Expected test exception.
        """
        token = None
        with pytest.raises(TranscriptionCancelledError):
            with cancellation_scope() as t:
                token = t
                raise TranscriptionCancelledError("Already cancelled")

        assert token is not None
        # Should not cancel since it's already a cancellation error
        assert token.cancelled is False

    def test_with_none_token__creates_new(self) -> None:
        """Test that passing None explicitly creates a new token."""
        with cancellation_scope(None) as token:
            assert isinstance(token, CancellationToken)
            assert token.cancelled is False
