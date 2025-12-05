"""Tests for benchmarks/collector.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from insanely_fast_whisper_api.benchmarks.collector import (
    BenchmarkCollector,
    GpuUtilSampler,
)


def test_benchmark_collector__init__creates_output_dir(tmp_path: Path) -> None:
    """Initialize BenchmarkCollector and create output directory."""
    output_dir = tmp_path / "benchmarks"
    collector = BenchmarkCollector(output_dir=output_dir)

    assert collector.output_dir == output_dir
    assert output_dir.exists()


def test_benchmark_collector__collect__writes_benchmark_file(tmp_path: Path) -> None:
    """Collect benchmark data and write to JSON file."""
    collector = BenchmarkCollector(output_dir=tmp_path)

    result_path = collector.collect(
        audio_path="/path/to/audio.mp3",
        task="transcribe",
        config={"model": "whisper-tiny"},
        runtime_seconds=10.5,
        total_time=12.0,
    )

    assert result_path.exists()
    assert result_path.suffix == ".json"
    assert "audio" in result_path.name
    assert "transcribe" in result_path.name

    # Verify JSON content
    data = json.loads(result_path.read_text())
    assert data["audio_path"] == "/path/to/audio.mp3"
    assert data["task"] == "transcribe"
    assert data["config"] == {"model": "whisper-tiny"}
    assert data["runtime_seconds"] == 10.5
    assert data["total_time_seconds"] == 12.0
    assert "recorded_at" in data


def test_benchmark_collector__collect__includes_extra_metadata(
    tmp_path: Path,
) -> None:
    """Include extra metadata in benchmark record."""
    collector = BenchmarkCollector(output_dir=tmp_path)

    result_path = collector.collect(
        audio_path="/path/to/audio.mp3",
        task="translate",
        config=None,
        runtime_seconds=None,
        total_time=15.0,
        extra={"custom_key": "custom_value"},
    )

    data = json.loads(result_path.read_text())
    assert data["extra"] == {"custom_key": "custom_value"}
    assert data["config"] == {}
    assert data["runtime_seconds"] is None


def test_benchmark_collector__collect__includes_gpu_stats(tmp_path: Path) -> None:
    """Include GPU stats in benchmark record."""
    collector = BenchmarkCollector(output_dir=tmp_path)

    gpu_stats = {
        "avg_gpu_load_percent": 75.5,
        "max_vram_mb": 8192,
    }

    result_path = collector.collect(
        audio_path="/path/to/audio.mp3",
        task="transcribe",
        config={},
        runtime_seconds=10.0,
        total_time=12.0,
        gpu_stats=gpu_stats,
    )

    data = json.loads(result_path.read_text())
    assert data["gpu_stats"] == gpu_stats


def test_benchmark_collector__collect__includes_format_quality(
    tmp_path: Path,
) -> None:
    """Include format quality metrics in benchmark record."""
    collector = BenchmarkCollector(output_dir=tmp_path)

    format_quality = {
        "srt": {"score": 0.95, "details": {"avg_cps": 18.5}},
    }

    result_path = collector.collect(
        audio_path="/path/to/audio.mp3",
        task="transcribe",
        config={},
        runtime_seconds=10.0,
        total_time=12.0,
        format_quality=format_quality,
    )

    data = json.loads(result_path.read_text())
    assert data["format_quality"] == format_quality


def test_benchmark_collector__collect__handles_zone_info_not_found(
    tmp_path: Path,
) -> None:
    """Fallback to UTC when APP_TIMEZONE is not found."""
    collector = BenchmarkCollector(output_dir=tmp_path)

    with patch(
        "insanely_fast_whisper_api.benchmarks.collector.APP_TIMEZONE",
        "Invalid/Timezone",
    ):
        result_path = collector.collect(
            audio_path="/path/to/audio.mp3",
            task="transcribe",
            config={},
            runtime_seconds=10.0,
            total_time=12.0,
        )

    # Should not raise and should create file
    assert result_path.exists()
    data = json.loads(result_path.read_text())
    assert "recorded_at" in data


def test_benchmark_collector__slugify__sanitizes_filename() -> None:
    """Slugify converts unsafe characters to safe filename."""
    assert BenchmarkCollector._slugify("hello world!") == "hello-world"
    assert BenchmarkCollector._slugify("file@#$name") == "file-name"
    assert BenchmarkCollector._slugify("my/path\\test") == "my-path-test"
    assert BenchmarkCollector._slugify("---test___") == "test"
    assert BenchmarkCollector._slugify("") == "benchmark"
    assert BenchmarkCollector._slugify("...") == "benchmark"


def test_gpu_util_sampler__init() -> None:
    """Initialize GpuUtilSampler with interval."""
    sampler = GpuUtilSampler(interval=1.0)
    assert sampler._interval == 1.0
    assert sampler._samples == []


def test_gpu_util_sampler__start__returns_false_when_pyamdgpuinfo_unavailable() -> None:
    """Return False when pyamdgpuinfo is not available."""
    sampler = GpuUtilSampler()

    with patch("insanely_fast_whisper_api.benchmarks.collector.pyamdgpuinfo", None):
        assert sampler.start() is False


def test_gpu_util_sampler__start__returns_false_on_gpu_error() -> None:
    """Return False when GPU initialization fails."""
    sampler = GpuUtilSampler()

    mock_pyamdgpuinfo = Mock()
    mock_pyamdgpuinfo.get_gpu.side_effect = RuntimeError("No GPU found")

    with patch(
        "insanely_fast_whisper_api.benchmarks.collector.pyamdgpuinfo",
        mock_pyamdgpuinfo,
    ):
        assert sampler.start() is False


def test_gpu_util_sampler__stop__handles_no_thread_gracefully() -> None:
    """Stop gracefully when no thread is running."""
    sampler = GpuUtilSampler()
    # Should not raise
    sampler.stop()


def test_gpu_util_sampler__stop__waits_for_thread(tmp_path: Path) -> None:
    """Stop waits for background thread to finish."""
    sampler = GpuUtilSampler(interval=0.1)

    # Manually set up thread to simulate running state
    import threading

    sampler._stop_event = threading.Event()
    sampler._thread = threading.Thread(target=lambda: None, daemon=True)
    sampler._thread.start()

    sampler.stop()

    assert sampler._stop_event is None
    assert sampler._thread is None


def test_gpu_util_sampler__summary__returns_none_when_no_samples() -> None:
    """Return None when no samples were collected."""
    sampler = GpuUtilSampler()
    assert sampler.summary() is None


def test_gpu_util_sampler__summary__returns_aggregated_stats() -> None:
    """Return aggregated GPU statistics from samples."""
    sampler = GpuUtilSampler()

    # Manually add samples: (load, vram_bytes)
    sampler._samples = [
        (0.5, 1024 * 1024 * 1024),  # 50% load, 1GB VRAM
        (0.7, 2 * 1024 * 1024 * 1024),  # 70% load, 2GB VRAM
        (0.6, 1.5 * 1024 * 1024 * 1024),  # 60% load, 1.5GB VRAM
    ]

    summary = sampler.summary()

    assert summary is not None
    assert summary["provider"] == "pyamdgpuinfo"
    assert summary["sample_count"] == 3
    assert summary["avg_gpu_load_percent"] == pytest.approx(60.0, rel=0.01)
    assert summary["max_gpu_load_percent"] == pytest.approx(70.0, rel=0.01)
    assert summary["avg_vram_mb"] == pytest.approx(1536.0, rel=0.01)
    assert summary["max_vram_mb"] == pytest.approx(2048.0, rel=0.01)


def test_gpu_util_sampler__run_loop__returns_early_if_no_gpu() -> None:
    """_run_loop returns early when GPU is not initialized."""
    sampler = GpuUtilSampler()
    sampler._gpu = None
    sampler._stop_event = Mock()

    # Should return immediately without waiting
    sampler._run_loop()

    # No samples should be collected
    assert len(sampler._samples) == 0
