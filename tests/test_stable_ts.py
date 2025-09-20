"""Tests for stable_ts integration wrapper."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

MODULE_PATH = (
    Path(__file__).parent.parent
    / "insanely_fast_whisper_api/core/integrations/stable_ts.py"
)

spec = importlib.util.spec_from_file_location("stable_ts", str(MODULE_PATH))
st = importlib.util.module_from_spec(spec)  # type: ignore
assert spec.loader is not None  # mypy guard
spec.loader.exec_module(st)  # type: ignore


@pytest.fixture(name="sample_result")
def _sample_result(tmp_path: Path) -> dict[str, Any]:
    """Return a minimal transcription-like dict with out-of-order timestamps."""
    return {
        "text": "hello world",
        "chunks": [
            {"text": "world", "timestamp": [2.0, 3.0]},
            {"text": "hello", "timestamp": [0.0, 1.0]},
        ],
        "original_file": str(Path(__file__).parent / "conversion-test-file.mp3"),
    }


def test_convert_to_stable_segments(sample_result: dict[str, Any]) -> None:
    """_convert_to_stable should rename *chunks* to *segments* and sort them."""
    converted = st._convert_to_stable(sample_result)  # pylint: disable=protected-access

    # Expect segments key, not chunks
    assert "segments" in converted and "chunks" not in converted
    segments = converted["segments"]
    assert segments[0]["text"] == "hello"
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 1.0
    assert segments[1]["text"] == "world"


def test_stabilize_returns_original_without_dependency(sample_result: dict[str, Any]) -> None:
    """If *stable_whisper* is unavailable, result should be returned unchanged."""
    # Force *stable_whisper* to None and reload to simulate missing dependency
    st.stable_whisper = None  # type: ignore
    result = st.stabilize_timestamps(sample_result)
    assert result == sample_result


def test_stabilize_success(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """When *stable_whisper* is present, timestamps should be stabilised and flag set."""

    # Stub for stable_whisper with transcribe_any returning word-level segments
    def fake_transcribe_any(
        inference_func: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:  # noqa: D401, D403
        base = inference_func()
        # Ensure conversion produced *segments*
        base.setdefault("segments", [])
        base["segments"].append({"text": "!", "start": 3.0, "end": 3.5})
        return base

    mock_sw = SimpleNamespace(transcribe_any=fake_transcribe_any)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)

    stabilised = st.stabilize_timestamps(sample_result)

    assert stabilised is not sample_result
    assert stabilised.get("stabilized") is True
    assert "segments" in stabilised and not stabilised.get("chunks")
    # Added segment by fake_transcribe_any should be present
    assert any(seg["text"] == "!" for seg in stabilised["segments"])
