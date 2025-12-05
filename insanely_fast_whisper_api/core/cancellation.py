"""Utilities for cooperative cancellation of long-running operations."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType

from insanely_fast_whisper_api.core.errors import TranscriptionCancelledError


class CancellationToken:
    """Signal that can be shared across threads to cancel work cooperatively."""

    def __init__(self) -> None:
        """Initialize the token in a non-cancelled state."""
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal that the associated work should stop."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Return True when cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise ``TranscriptionCancelledError`` if cancellation was requested.

        Raises:
            TranscriptionCancelledError: If ``cancel()`` was triggered earlier.
        """
        if self.cancelled:
            raise TranscriptionCancelledError("Transcription cancelled by caller")


@dataclass(slots=True)
class CancellationScope:
    """Context manager that cancels a token when exiting the scope early."""

    token: CancellationToken

    def cancel(self) -> None:
        """Convenience method mirroring ``CancellationToken.cancel``."""
        self.token.cancel()

    def __enter__(self) -> CancellationToken:
        """Return the managed token."""
        return self.token

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Cancel on unexpected errors while preserving exception propagation.

        Args:
            exc_type: The exception type raised inside the scope, if any.
            exc: The exception instance raised inside the scope, if any.
            traceback: Associated traceback for ``exc``.

        Returns:
            False so that exceptions continue to bubble up.
        """
        if exc_type is not None and not issubclass(
            exc_type, TranscriptionCancelledError
        ):
            # Cancelling the token ensures downstream tasks also wind down.
            self.token.cancel()
        # Never suppress exceptions; cooperative cancellation relies on propagation.
        return False


@contextlib.contextmanager
def cancellation_scope(
    token: CancellationToken | None = None,
) -> Iterator[CancellationToken]:
    """Context manager returning a token and cancelling it on unexpected errors.

    Args:
        token: Existing token to reuse; when omitted a new token is created.

    Yields:
        CancellationToken: The token governing the enclosed block.
    """
    effective_token = token or CancellationToken()
    scope = CancellationScope(effective_token)
    with scope:
        yield effective_token
