"""Tests for the tqdm-based progress reporter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from insanely_fast_whisper_api.cli.progress_tqdm import TqdmProgressReporter


class _FakeBar:
    """Simple stand-in for tqdm bars collecting lifecycle calls."""

    def __init__(self, *, total: int | None = None, **_: object) -> None:
        self.total = total
        self.n = 0
        self.closed = False

    def update(self, value: int) -> None:
        self.n += value

    def refresh(self) -> None:  # noqa: D401 - method exists for compatibility
        """No-op refresh for compatibility."""
        return

    def close(self) -> None:
        self.closed = True


class _FakeTqdmModule:
    """Callable shim mimicking the tqdm API used by the reporter."""

    def __init__(self) -> None:
        self.bars: list[_FakeBar] = []
        self.write_calls: list[str] = []

    def __call__(self, *args: object, **kwargs: object) -> _FakeBar:
        bar = _FakeBar(total=kwargs.get("total"))
        self.bars.append(bar)
        return bar

    def write(self, message: str) -> None:
        self.write_calls.append(message)


@pytest.fixture
def fake_tqdm(monkeypatch: pytest.MonkeyPatch) -> _FakeTqdmModule:
    """Provide a patched tqdm shim for the duration of a test.

    Returns:
        _FakeTqdmModule: Module-like object capturing tqdm interactions.
    """
    fake_module = _FakeTqdmModule()
    monkeypatch.setattr(
        "insanely_fast_whisper_api.cli.progress_tqdm.tqdm", fake_module
    )
    return fake_module


def test_tqdm_reporter_full_flow(fake_tqdm: _FakeTqdmModule) -> None:
    """Exercise the happy-path flow with progress enabled."""
    reporter = TqdmProgressReporter(enabled=True)

    reporter.on_model_load_finished()
    reporter.on_audio_loading_finished(duration_sec=None)
    reporter.on_chunking_started(total_chunks=2)
    reporter.on_chunk_done(index=0)
    reporter.on_chunk_done(index=1)
    reporter.on_postprocess_finished(name="demucs")
    reporter.on_postprocess_finished(name="vad threshold=0.35")
    reporter.on_export_started(total_items=2)
    reporter.on_export_item_done(index=0, label="JSON::/tmp/output.json")
    reporter.on_export_item_done(index=1, label="TXT::/tmp/output.txt")
    reporter.on_completed()

    assert fake_tqdm.write_calls == [
        "\u2714 Model loaded",
        "\u2714 Audio prepared.",
        "\u2714 Transcription",
        "\u2714 Demucs vocals isolated",
        "\u2714 VAD applied (threshold=0.35)",
    ]
    assert all(bar.closed for bar in fake_tqdm.bars)


def test_tqdm_reporter_disabled_is_noop(fake_tqdm: _FakeTqdmModule) -> None:
    """Ensure disabled reporter performs no tqdm calls."""
    reporter = TqdmProgressReporter(enabled=False)

    reporter.on_model_load_finished()
    reporter.on_audio_loading_finished(duration_sec=None)
    reporter.on_chunking_started(total_chunks=3)
    reporter.on_chunk_done(index=0)
    reporter.on_postprocess_finished(name="demucs")
    reporter.on_export_started(total_items=1)
    reporter.on_export_item_done(index=0, label="JSON::/tmp/output.json")
    reporter.on_error(message="boom")

    assert fake_tqdm.bars == []


def test_tqdm_reporter_zero_chunks_no_bar(fake_tqdm: _FakeTqdmModule) -> None:
    """Test that zero total chunks doesn't create a progress bar."""
    reporter = TqdmProgressReporter(enabled=True)

    # Zero chunks should not create a bar
    reporter.on_chunking_started(total_chunks=0)
    assert len(fake_tqdm.bars) == 0

    # None chunks should not create a bar
    reporter.on_chunking_started(total_chunks=None)
    assert len(fake_tqdm.bars) == 0


def test_tqdm_reporter_chunk_done_early_completion(fake_tqdm: _FakeTqdmModule) -> None:
    """Test chunk completion when bar finishes early."""
    reporter = TqdmProgressReporter(enabled=True)

    # Start with 2 chunks
    reporter.on_chunking_started(total_chunks=2)
    assert len(fake_tqdm.bars) == 1
    bar = fake_tqdm.bars[0]
    assert bar.total == 2

    # Complete both chunks
    reporter.on_chunk_done(index=0)
    assert bar.n == 1
    reporter.on_chunk_done(index=1)
    assert bar.n == 2  # Should be set to total
    assert bar.closed

    # Check completion message
    assert "✔ Transcription" in fake_tqdm.write_calls[-1]


def test_tqdm_reporter_inference_noops(fake_tqdm: _FakeTqdmModule) -> None:
    """Test that inference callbacks are no-ops."""
    reporter = TqdmProgressReporter(enabled=True)

    # These should do nothing
    reporter.on_inference_started(total_batches=10)
    reporter.on_inference_batch_done(index=0)

    assert len(fake_tqdm.bars) == 0
    assert fake_tqdm.write_calls == []


def test_tqdm_reporter_postprocess_started_noop(fake_tqdm: _FakeTqdmModule) -> None:
    """Test that postprocess started is a no-op."""
    reporter = TqdmProgressReporter(enabled=True)

    reporter.on_postprocess_started("any_name")
    assert fake_tqdm.write_calls == []


def test_tqdm_reporter_postprocess_vad_parsing(fake_tqdm: _FakeTqdmModule) -> None:
    """Test VAD threshold parsing in postprocess finished."""
    reporter = TqdmProgressReporter(enabled=True)

    # Test VAD with threshold
    reporter.on_postprocess_finished("vad threshold=0.35")
    assert "✔ VAD applied (threshold=0.35)" in fake_tqdm.write_calls

    # Test VAD without threshold
    reporter.on_postprocess_finished("vad")
    assert "✔ VAD applied" in fake_tqdm.write_calls[-1]

    # Test malformed VAD
    reporter.on_postprocess_finished("vad threshold=")
    assert "✔ VAD applied" in fake_tqdm.write_calls[-1]


def test_tqdm_reporter_postprocess_generic(fake_tqdm: _FakeTqdmModule) -> None:
    """Test generic postprocess finished message."""
    reporter = TqdmProgressReporter(enabled=True)

    reporter.on_postprocess_finished("unknown_process")
    assert "✔ Post completed" in fake_tqdm.write_calls[-1]


def test_tqdm_reporter_export_single_item_no_bar(fake_tqdm: _FakeTqdmModule) -> None:
    """Test single export item doesn't create progress bar."""
    reporter = TqdmProgressReporter(enabled=True)

    reporter.on_export_started(total_items=1)
    assert len(fake_tqdm.bars) == 0

    reporter.on_export_item_done(index=0, label="JSON::/tmp/test.json")
    # Should not create any bar or messages for single items
    assert len(fake_tqdm.bars) == 0


def test_tqdm_reporter_export_multi_item_with_bar(fake_tqdm: _FakeTqdmModule) -> None:
    """Test multiple export items create and update progress bar."""
    reporter = TqdmProgressReporter(enabled=True)

    reporter.on_export_started(total_items=3)
    assert len(fake_tqdm.bars) == 1
    bar = fake_tqdm.bars[0]
    assert bar.total == 3

    # Complete items
    reporter.on_export_item_done(index=0, label="JSON::/tmp/test.json")
    assert bar.n == 1
    reporter.on_export_item_done(index=1, label="TXT::/tmp/test.txt")
    assert bar.n == 2
    reporter.on_export_item_done(index=2, label="SRT::/tmp/test.srt")
    assert bar.n == 3
    assert bar.closed


def test_tqdm_reporter_completion_error_handling(fake_tqdm: _FakeTqdmModule) -> None:
    """Test error handling in completion cleanup."""
    reporter = TqdmProgressReporter(enabled=True)

    # Create a bar that will fail to close
    reporter.on_chunking_started(total_chunks=1)

    # Mock the bar to raise exception on close
    bar = fake_tqdm.bars[0]
    bar.close = Mock(side_effect=Exception("Close failed"))

    # Completion should handle errors gracefully
    reporter.on_completed()
    # Bar should still be attempted to close despite error
    bar.close.assert_called_once()
