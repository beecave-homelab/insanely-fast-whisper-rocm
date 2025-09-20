"""Tests for the tqdm-based progress reporter."""

from __future__ import annotations

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
    """Provide a patched tqdm shim for the duration of a test."""

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

    assert fake_tqdm.write_calls == []
    assert fake_tqdm.bars == []
