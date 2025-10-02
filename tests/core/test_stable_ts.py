"""Tests for stable_ts integration wrapper."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "insanely_fast_whisper_api"
    / "core"
    / "integrations"
    / "stable_ts.py"
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
        "original_file": str(
            Path(__file__).parent / "data" / "conversion-test-file.mp3"
        ),
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


def test_stabilize_returns_original_without_dependency(
    sample_result: dict[str, Any],
) -> None:
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
        inference_func: Callable[[], dict[str, Any]],
        audio: str,
        denoiser: str | None = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        check_sorted: bool = False,
        **kwargs: dict[str, Any],
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


def test_to_dict_with_dict() -> None:
    """Test _to_dict with a plain dict (already covered but for completeness)."""
    result = {"text": "test", "segments": []}
    assert st._to_dict(result) == result


def test_to_dict_with_to_dict_method() -> None:
    """Test _to_dict with object having to_dict method."""

    class MockObject:
        def to_dict(self) -> dict[str, Any]:
            return {"mocked": True, "value": 42}

    obj = MockObject()
    result = st._to_dict(obj)
    assert result == {"mocked": True, "value": 42}


def test_to_dict_with_model_dump_method() -> None:
    """Test _to_dict with object having model_dump method."""

    class MockObject:
        def model_dump(self) -> dict[str, Any]:
            return {"model": "dumped", "data": [1, 2, 3]}

    obj = MockObject()
    result = st._to_dict(obj)
    assert result == {"model": "dumped", "data": [1, 2, 3]}


def test_to_dict_with_asdict_method() -> None:
    """Test _to_dict with object having _asdict method."""

    class MockObject:
        def _asdict(self) -> dict[str, Any]:
            return {"as": "dict", "converted": True}

    obj = MockObject()
    result = st._to_dict(obj)
    assert result == {"as": "dict", "converted": True}


def test_to_dict_fallback_to_str() -> None:
    """Test _to_dict fallback to str() for unknown objects."""
    obj = "some string object"
    result = st._to_dict(obj)
    assert result == {"text": "some string object"}


def test_convert_to_stable_segments_already_exist() -> None:
    """Test _convert_to_stable when segments already exist (no chunks rename)."""
    result = {
        "text": "test",
        "segments": [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 1.0, "end": 2.0},
        ],
    }
    converted = st._convert_to_stable(result)
    assert "segments" in converted
    assert "chunks" not in converted
    assert converted["segments"] == result["segments"]


def test_convert_to_stable_no_timestamp_conversion() -> None:
    """Test _convert_to_stable with segments that already have start/end fields."""
    result = {
        "text": "test",
        "chunks": [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 1.0, "end": 2.0},
        ],
    }
    converted = st._convert_to_stable(result)
    assert "segments" in converted
    assert "chunks" not in converted
    assert len(converted["segments"]) == 2


def test_convert_to_stable_wrong_timestamp_order() -> None:
    """Test _convert_to_stable with segments having wrong timestamp order."""
    result = {
        "text": "test",
        "chunks": [
            {"text": "hello", "start": 1.0, "end": 0.0},  # Wrong order
            {"text": "world", "start": 2.0, "end": 3.0},
        ],
    }
    converted = st._convert_to_stable(result)
    segments = converted["segments"]
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 1.0  # Swapped
    assert segments[1]["start"] == 2.0 and segments[1]["end"] == 3.0  # Unchanged


def test_convert_to_stable_none_timestamps() -> None:
    """Test _convert_to_stable with segments having None start/end values."""
    result = {
        "text": "test",
        "chunks": [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": None, "end": 2.0},  # None start
            {"text": "!", "start": 2.0, "end": None},  # None end
        ],
    }
    converted = st._convert_to_stable(result)
    segments = converted["segments"]
    # Only segments with valid timestamps should remain
    assert len(segments) == 1
    assert segments[0]["text"] == "hello"


def test_convert_to_stable_overlapping_segments() -> None:
    """Test _convert_to_stable with overlapping segments."""
    result = {
        "text": "test",
        "chunks": [
            {"text": "hello", "start": 0.0, "end": 2.0},
            {"text": "world", "start": 1.0, "end": 3.0},  # Overlaps with previous
            {"text": "!", "start": 4.0, "end": 5.0},  # No overlap
        ],
    }
    converted = st._convert_to_stable(result)
    segments = converted["segments"]
    assert len(segments) == 3
    # Check that overlapping segment was adjusted
    assert segments[1]["start"] >= segments[0]["end"]


def test_stabilize_missing_audio_path() -> None:
    """Test stabilize_timestamps with missing audio path."""
    result = {"text": "test"}  # No original_file or audio_file_path
    stabilized = st.stabilize_timestamps(result)
    assert stabilized == result  # Should return unchanged


def test_stabilize_nonexistent_audio_file(sample_result: dict[str, Any]) -> None:
    """Test stabilize_timestamps with non-existent audio file."""
    # Modify the path to a non-existent file
    result = sample_result.copy()
    result["original_file"] = "/non/existent/file.mp3"
    stabilized = st.stabilize_timestamps(result)
    assert stabilized == result  # Should return unchanged


def test_stabilize_transcribe_any_failure_with_postprocess(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps when transcribe_any fails but postprocess succeeds."""

    # Mock transcribe_any to fail
    def failing_transcribe_any(*args: object, **kwargs: object) -> None:
        raise Exception("transcribe_any failed")

    # Mock postprocess to succeed
    def mock_postprocess(
        converted: dict[str, Any], audio: str, **kwargs: object
    ) -> dict[str, Any]:
        return {
            "segments": [{"text": "processed", "start": 0.0, "end": 1.0}],
            "text": "processed result",
        }

    mock_sw = SimpleNamespace(transcribe_any=failing_transcribe_any)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(st, "_postprocess", mock_postprocess, raising=False)

    stabilized = st.stabilize_timestamps(sample_result)

    assert stabilized.get("stabilized") is True
    assert stabilized.get("stabilization_path") == "postprocess_word_timestamps"
    assert "segments" in stabilized


def test_stabilize_all_methods_fail(
    sample_result: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test stabilize_timestamps when both transcribe_any and postprocess fail."""

    # Mock transcribe_any to fail
    def failing_transcribe_any(*args: object, **kwargs: object) -> None:
        raise Exception("transcribe_any failed")

    # Mock postprocess to fail
    def failing_postprocess(*args: object, **kwargs: object) -> None:  # type: ignore
        raise Exception("postprocess failed")

    mock_sw = SimpleNamespace(transcribe_any=failing_transcribe_any)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(st, "_postprocess", failing_postprocess, raising=False)

    stabilized = st.stabilize_timestamps(sample_result)

    assert stabilized == sample_result  # Should return unchanged
    assert stabilized.get("stabilized") is None  # No stabilization flag set


def test_stabilize_segments_without_timestamps(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps when processed segments don't have timestamps."""

    def fake_transcribe_any(
        inference_func: Callable[[], dict[str, Any]],
        audio: str,
        denoiser: str | None = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        check_sorted: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        base = inference_func()
        # Return segments without start/end timestamps
        base["segments"] = [{"text": "no timestamps"}]
        return base

    mock_sw = SimpleNamespace(transcribe_any=fake_transcribe_any)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)

    stabilized = st.stabilize_timestamps(sample_result)

    # Should still be stabilized but keep original chunks since no timestamps
    assert stabilized.get("stabilized") is True
    assert "chunks" in stabilized  # Original chunks preserved
    assert "segments" in stabilized


def test_convert_to_stable_comprehensive_edge_cases() -> None:
    """Test _convert_to_stable with comprehensive edge cases."""
    result = {
        "text": "test",
        "chunks": [
            # Normal case
            {"text": "hello", "timestamp": [0.0, 1.0]},
            # Wrong order (should be swapped)
            {"text": "world", "timestamp": [2.0, 1.0]},
            # Already has start/end (should be preserved)
            {"text": "already", "start": 3.0, "end": 4.0},
            # None values (should be filtered out)
            {"text": "none_start", "timestamp": [None, 5.0]},
            {"text": "none_end", "timestamp": [5.0, None]},
            # Overlapping (should be adjusted)
            {"text": "overlap", "timestamp": [4.5, 6.0]},
        ],
    }
    converted = st._convert_to_stable(result)
    segments = converted["segments"]

    # Should have 4 segments (filtered out the None ones)
    assert len(segments) == 4

    # Check ordering and values
    assert segments[0]["text"] == "hello"
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 1.0

    assert segments[1]["text"] == "world"
    assert segments[1]["start"] == 1.0 and segments[1]["end"] == 2.0  # Swapped

    assert segments[2]["text"] == "already"
    assert segments[2]["start"] == 3.0 and segments[2]["end"] == 4.0

    assert segments[3]["text"] == "overlap"
    assert segments[3]["start"] >= segments[2]["end"]  # Adjusted for overlap


def test_stabilize_timestamps_comprehensive_error_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stabilize_timestamps with various error conditions."""
    # Test with audio_file_path instead of original_file
    result = {"text": "test", "audio_file_path": "/nonexistent.mp3"}
    stabilized = st.stabilize_timestamps(result)
    assert stabilized == result  # Should return unchanged

    # Test with empty string path
    result2 = {"text": "test", "original_file": ""}
    stabilized2 = st.stabilize_timestamps(result2)
    assert stabilized2 == result2  # Should return unchanged


def test_to_dict_comprehensive() -> None:
    """Test _to_dict with comprehensive object types."""

    # Test with various object types that have conversion methods
    class ToDictObj:
        def to_dict(self) -> dict[str, Any]:
            return {"type": "to_dict", "value": 1}

    class ModelDumpObj:
        def model_dump(self) -> dict[str, Any]:
            return {"type": "model_dump", "value": 2}

    class AsDictObj:
        def _asdict(self) -> dict[str, Any]:
            return {"type": "_asdict", "value": 3}

    class NoMethodsObj:
        def __init__(self) -> None:
            self.value = 42

    # Test all conversion methods
    assert st._to_dict(ToDictObj()) == {"type": "to_dict", "value": 1}
    assert st._to_dict(ModelDumpObj()) == {"type": "model_dump", "value": 2}
    assert st._to_dict(AsDictObj()) == {"type": "_asdict", "value": 3}
    # Test fallback to str() for objects without conversion methods
    no_methods_result = st._to_dict(NoMethodsObj())
    assert "text" in no_methods_result
    assert "NoMethodsObj object at" in no_methods_result["text"]
    assert st._to_dict(42) == {"text": "42"}
    assert st._to_dict([1, 2, 3]) == {"text": "[1, 2, 3]"}


def test_stabilize_with_progress_callback_unavailable() -> None:
    """Test stabilize_timestamps with progress_cb when stable_whisper unavailable."""
    result = {"text": "test", "original_file": "/test.mp3"}
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    # Force stable_whisper to None
    st.stable_whisper = None  # type: ignore
    stabilized = st.stabilize_timestamps(result, progress_cb=progress_cb)

    assert stabilized == result
    assert any("unavailable" in msg for msg in messages)


def test_stabilize_with_progress_callback_missing_path() -> None:
    """Test stabilize_timestamps with progress_cb when audio path missing."""
    result = {"text": "test"}  # No audio path
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    # Ensure stable_whisper is available (mock it)
    mock_sw = SimpleNamespace()
    st.stable_whisper = mock_sw  # type: ignore

    stabilized = st.stabilize_timestamps(result, progress_cb=progress_cb)

    assert stabilized == result
    assert any("missing" in msg for msg in messages)


def test_stabilize_with_progress_callback_nonexistent_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stabilize_timestamps with progress_cb when audio file doesn't exist."""
    result = {"text": "test", "original_file": "/absolutely/nonexistent/path.mp3"}
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    # Ensure stable_whisper is available
    mock_sw = SimpleNamespace()
    st.stable_whisper = mock_sw  # type: ignore

    # Ensure SKIP_FS_CHECKS is False so the file check is enforced
    from insanely_fast_whisper_api.utils import constants

    monkeypatch.setattr(constants, "SKIP_FS_CHECKS", False)

    stabilized = st.stabilize_timestamps(result, progress_cb=progress_cb)

    assert stabilized == result
    assert any("not found" in msg for msg in messages)


def test_stabilize_with_progress_callback_success_postprocess(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps with progress_cb during successful postprocess."""
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    def mock_postprocess(
        converted: dict[str, Any], audio: str, **kwargs: object
    ) -> dict[str, Any]:
        return {
            "segments": [{"text": "processed", "start": 0.0, "end": 1.0}],
            "text": "processed result",
        }

    mock_sw = SimpleNamespace(transcribe_any=lambda *a, **k: None)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(st, "_postprocess", mock_postprocess, raising=False)

    stabilized = st.stabilize_timestamps(sample_result, progress_cb=progress_cb)

    assert stabilized.get("stabilized") is True
    assert len(messages) > 0
    assert any("running" in msg for msg in messages)
    assert any("successful" in msg for msg in messages)


def test_stabilize_with_progress_callback_success_lambda(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps with progress_cb during successful lambda path."""
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    def fake_transcribe_any(
        inference_func: Callable[[], dict[str, Any]],
        audio: str,
        denoiser: str | None = None,
        vad: bool = False,
        vad_threshold: float = 0.35,
        check_sorted: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        base = inference_func()
        base.setdefault("segments", [])
        base["segments"].append({"text": "!", "start": 3.0, "end": 3.5})
        return base

    mock_sw = SimpleNamespace(transcribe_any=fake_transcribe_any)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(st, "_postprocess", None, raising=False)
    monkeypatch.setattr(st, "_postprocess_alt", None, raising=False)

    stabilized = st.stabilize_timestamps(
        sample_result, demucs=True, vad=True, vad_threshold=0.4, progress_cb=progress_cb
    )

    assert stabilized.get("stabilized") is True
    assert len(messages) > 0
    assert any("demucs=True" in msg for msg in messages)
    assert any("successful" in msg for msg in messages)


def test_stabilize_with_progress_callback_all_fail(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps with progress_cb when all methods fail."""
    messages: list[str] = []

    def progress_cb(msg: str) -> None:
        messages.append(msg)

    def failing_func(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Failed")

    mock_sw = SimpleNamespace(transcribe_any=failing_func)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(st, "_postprocess", failing_func, raising=False)

    stabilized = st.stabilize_timestamps(sample_result, progress_cb=progress_cb)

    assert stabilized == sample_result
    assert len(messages) > 0
    assert any("failed" in msg for msg in messages)


def test_stabilize_postprocess_typeerror_fallback(
    monkeypatch: pytest.MonkeyPatch, sample_result: dict[str, Any]
) -> None:
    """Test stabilize_timestamps when postprocess raises TypeError and falls back."""
    call_count = {"count": 0}

    def mock_postprocess_with_typeerror(
        converted: dict[str, Any], audio: str, **kwargs: object
    ) -> dict[str, Any]:
        call_count["count"] += 1
        if call_count["count"] == 1:
            # First call with extra kwargs raises TypeError
            raise TypeError("Unexpected keyword argument")
        # Second call with minimal args succeeds
        return {
            "segments": [{"text": "fallback", "start": 0.0, "end": 1.0}],
            "text": "fallback result",
        }

    mock_sw = SimpleNamespace(transcribe_any=lambda *a, **k: None)
    monkeypatch.setattr(st, "stable_whisper", mock_sw, raising=False)
    monkeypatch.setattr(
        st, "_postprocess", mock_postprocess_with_typeerror, raising=False
    )

    stabilized = st.stabilize_timestamps(sample_result, demucs=True, vad=True)

    assert stabilized.get("stabilized") is True
    assert call_count["count"] == 2  # Called twice due to TypeError fallback
    assert stabilized.get("stabilization_path") == "postprocess_word_timestamps"
