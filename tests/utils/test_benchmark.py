"""Tests for insanely_fast_whisper_rocm.utils.benchmark module.

This module contains tests for benchmark collection and metric gathering.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.utils.benchmark import (
    BenchmarkCollector,
    BenchmarkResult,
)


class TestBenchmarkResult:
    """Test suite for BenchmarkResult model."""

    def test_benchmark_result__minimal_creation(self) -> None:
        """Test creating BenchmarkResult with minimal required fields."""
        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            model="openai/whisper-tiny",
            device="cpu",
            runtime_seconds=10.5,
            total_wall_time_seconds=15.0,
            system={"os": "Linux", "python_version": "3.10"},
        )
        assert result.timestamp == "2024-01-01T00:00:00"
        assert result.model == "openai/whisper-tiny"
        assert result.device == "cpu"
        assert result.runtime_seconds == 10.5
        assert result.system["os"] == "Linux"

    def test_benchmark_result__with_gpu_info(self) -> None:
        """Test creating BenchmarkResult with GPU information."""
        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            model="openai/whisper-small",
            device="cuda",
            runtime_seconds=5.0,
            total_wall_time_seconds=8.0,
            system={"os": "Linux"},
            gpu={"name": "NVIDIA RTX 3090", "total_vram_mb": 24000},
        )
        assert result.gpu is not None
        assert result.gpu["name"] == "NVIDIA RTX 3090"

    def test_benchmark_result__with_memory_info(self) -> None:
        """Test creating BenchmarkResult with memory information."""
        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            model=None,
            device=None,
            runtime_seconds=10.0,
            total_wall_time_seconds=12.0,
            system={},
            memory={"ram_total_mb": 16000, "ram_average_mb": 8000},
        )
        assert result.memory is not None
        assert result.memory["ram_total_mb"] == 16000

    def test_benchmark_result__frozen_config(self) -> None:
        """Test that BenchmarkResult is frozen (immutable)."""
        result = BenchmarkResult(
            timestamp="2024-01-01T00:00:00",
            model="test",
            device="cpu",
            runtime_seconds=1.0,
            total_wall_time_seconds=2.0,
            system={},
        )
        # Pydantic v1 and v2 both support frozen models
        with pytest.raises(Exception):  # ValidationError or AttributeError
            result.model = "new_model"  # type: ignore[misc]


class TestBenchmarkCollector:
    """Test suite for BenchmarkCollector class."""

    def test_benchmark_collector__initialization(self) -> None:
        """Test BenchmarkCollector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            assert Path(tmpdir).exists()
            assert collector._bench_dir == Path(tmpdir)

    def test_benchmark_collector__creates_directory(self) -> None:
        """Test that BenchmarkCollector creates the benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir) / "benchmarks"
            assert not bench_dir.exists()
            BenchmarkCollector(benchmarks_dir=bench_dir)
            assert bench_dir.exists()

    def test_benchmark_collector__start_records_time(self) -> None:
        """Test that start() records the start time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            assert collector._start_time is None
            collector.start()
            assert collector._start_time is not None
            collector.stop_sampling()

    def test_benchmark_collector__stop_returns_elapsed_time(self) -> None:
        """Test that stop() returns elapsed time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.start()
            time.sleep(0.1)  # Small delay
            elapsed = collector.stop()
            collector.stop_sampling()
            assert elapsed >= 0.1

    def test_benchmark_collector__stop_before_start_raises(self) -> None:
        """Test that stop() raises RuntimeError if called before start()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            with pytest.raises(RuntimeError, match="stop.*before start"):
                collector.stop()

    def test_benchmark_collector__set_model_load_time(self) -> None:
        """Test setting model load time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.set_model_load_time(5.5)
            assert collector._model_load_time == 5.5

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil")
    def test_collect_system_metrics__with_psutil(self, mock_psutil: MagicMock) -> None:
        """Test collecting system metrics when psutil is available."""
        mock_vm = MagicMock()
        mock_vm.total = 16 * 1024**3  # 16 GB
        mock_vm.used = 8 * 1024**3  # 8 GB
        mock_psutil.virtual_memory.return_value = mock_vm

        metrics = BenchmarkCollector._collect_system_metrics()

        assert "os" in metrics
        assert "python_version" in metrics
        assert metrics["ram_total_mb"] == 16384.0
        assert metrics["ram_used_mb"] == 8192.0

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil", None)
    def test_collect_system_metrics__without_psutil(self) -> None:
        """Test collecting system metrics when psutil is unavailable."""
        metrics = BenchmarkCollector._collect_system_metrics()

        assert "os" in metrics
        assert "python_version" in metrics
        assert "ram_total_mb" not in metrics
        assert "ram_used_mb" not in metrics

    @patch("insanely_fast_whisper_rocm.utils.benchmark.torch")
    def test_collect_gpu_metrics__cuda_available(self, mock_torch: MagicMock) -> None:
        """Test collecting GPU metrics when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3090"

        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 12 * 1024**3  # 12 GB

        metrics = BenchmarkCollector._collect_gpu_metrics()

        assert metrics is not None
        assert metrics["name"] == "NVIDIA RTX 3090"
        assert metrics["total_vram_mb"] == 24576.0
        assert metrics["vram_used_mb"] == 12288.0

    @patch("insanely_fast_whisper_rocm.utils.benchmark.torch", None)
    @patch("insanely_fast_whisper_rocm.utils.benchmark.pyamdgpuinfo", None)
    def test_collect_gpu_metrics__no_gpu_library(self) -> None:
        """Test collecting GPU metrics when no GPU library is available."""
        metrics = BenchmarkCollector._collect_gpu_metrics()
        assert metrics is None

    def test_stop_sampling__stops_thread(self) -> None:
        """Test that stop_sampling stops the sampling thread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.start()
            time.sleep(0.2)
            collector.stop_sampling()
            # Thread should be stopped
            assert collector._stop_event is not None
            assert collector._stop_event.is_set()

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil")
    @patch("insanely_fast_whisper_rocm.utils.benchmark.torch", None)
    def test_collect__writes_json_file(self, mock_psutil: MagicMock) -> None:
        """Test that collect() writes a JSON file."""
        mock_vm = MagicMock()
        mock_vm.total = 16 * 1024**3
        mock_vm.used = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vm

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.start()
            time.sleep(0.1)
            elapsed = collector.stop()

            path = collector.collect(
                audio_path="test_audio.wav",
                task="transcribe",
                config={"model": "openai/whisper-tiny", "device": "cpu"},
                runtime_seconds=10.0,
                total_time=elapsed,
            )

            assert path.exists()
            assert path.suffix == ".json"
            assert path.name.startswith("benchmark_")

            # Verify JSON content
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                assert data["model"] == "openai/whisper-tiny"
                assert data["device"] == "cpu"
                assert data["runtime_seconds"] == 10.0
                assert "system" in data

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil")
    def test_collect__with_extra_data(self, mock_psutil: MagicMock) -> None:
        """Test that collect() includes extra data."""
        mock_vm = MagicMock()
        mock_vm.total = 16 * 1024**3
        mock_vm.used = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vm

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.start()
            time.sleep(0.05)
            elapsed = collector.stop()

            extra_data = {"custom_metric": 42, "note": "test run"}
            path = collector.collect(
                audio_path="test.wav",
                task="translate",
                config={"model": "whisper-base"},
                runtime_seconds=5.0,
                total_time=elapsed,
                extra=extra_data,
            )

            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                assert data["extra"] == extra_data

    def test_average_metrics__empty_samples(self) -> None:
        """Test _average_metrics with no samples collected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            sys_avg, gpu_avg = collector._average_metrics()
            # Should return current metrics
            assert "os" in sys_avg
            assert "python_version" in sys_avg

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil")
    def test_average_metrics__with_samples(self, mock_psutil: MagicMock) -> None:
        """Test _average_metrics with collected samples."""
        mock_vm = MagicMock()
        mock_vm.total = 16 * 1024**3
        mock_vm.used = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vm

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            # Manually add samples
            collector._samples = [
                {"system": {"ram_used_mb": 1000}, "gpu": None},
                {"system": {"ram_used_mb": 2000}, "gpu": None},
                {"system": {"ram_used_mb": 3000}, "gpu": None},
            ]
            sys_avg, gpu_avg = collector._average_metrics()
            assert sys_avg["ram_used_mb"] == 2000.0

    def test_gpu_vram_stats__no_samples(self) -> None:
        """Test _gpu_vram_stats with no samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            min_val, max_val, med_val = collector._gpu_vram_stats()
            assert min_val is None
            assert max_val is None
            assert med_val is None

    def test_gpu_vram_stats__with_samples(self) -> None:
        """Test _gpu_vram_stats with valid samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector._samples = [
                {"gpu": {"vram_used_mb": 1000}},
                {"gpu": {"vram_used_mb": 2000}},
                {"gpu": {"vram_used_mb": 3000}},
                {"gpu": {"vram_used_mb": 0}},  # Should be ignored
            ]
            min_val, max_val, med_val = collector._gpu_vram_stats()
            assert min_val == 1000.0
            assert max_val == 3000.0
            assert med_val == 2000.0

    def test_gpu_vram_stats__ignores_zero_values(self) -> None:
        """Test that _gpu_vram_stats ignores zero VRAM values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector._samples = [
                {"gpu": {"vram_used_mb": 0}},
                {"gpu": {"vram_used_mb": 0}},
            ]
            min_val, max_val, med_val = collector._gpu_vram_stats()
            assert min_val is None
            assert max_val is None
            assert med_val is None

    @patch("insanely_fast_whisper_rocm.utils.benchmark.psutil")
    def test_collect__handles_invalid_task(self, mock_psutil: MagicMock) -> None:
        """Test that collect() handles invalid task values gracefully."""
        mock_vm = MagicMock()
        mock_vm.total = 16 * 1024**3
        mock_vm.used = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vm

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = BenchmarkCollector(benchmarks_dir=tmpdir)
            collector.start()
            time.sleep(0.05)
            elapsed = collector.stop()

            # Should fallback to TRANSCRIBE for invalid task
            path = collector.collect(
                audio_path="test.wav",
                task="invalid_task",
                config={},
                runtime_seconds=1.0,
                total_time=elapsed,
            )
            assert path.exists()
