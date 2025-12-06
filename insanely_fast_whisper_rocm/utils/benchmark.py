"""Benchmark utilities for collecting and persisting performance metrics.

This module is *optional* – it only incurs extra dependencies (`psutil`,
`pyamdgpuinfo`, `pydantic`) and minor overhead when the CLI is executed with
`--benchmark`. When the flag is absent, importing this module is avoided to
keep startup time minimal.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from insanely_fast_whisper_rocm.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)


try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import pyamdgpuinfo  # type: ignore
except ImportError:  # pragma: no cover
    pyamdgpuinfo = None  # type: ignore

try:
    from pydantic import BaseModel, ConfigDict  # type: ignore
except ImportError:  # pragma: no cover
    # The core package already depends on pydantic; but guard just in case
    BaseModel = object  # type: ignore[assignment]
    ConfigDict = dict  # type: ignore[misc,assignment]

__all__ = [
    "BenchmarkResult",
    "BenchmarkCollector",
]


class BenchmarkResult(BaseModel):  # type: ignore[misc]
    """Schema for benchmark JSON file."""

    timestamp: str  # ISO8601
    model: str | None
    device: str | None
    batch_size: int | None = None
    runtime_seconds: float | None
    total_wall_time_seconds: float | None
    model_load_seconds: float | None = None

    system: dict[str, Any]
    gpu: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None

    extra: dict[str, Any] | None = None

    model_config: ConfigDict = ConfigDict(frozen=True)


class BenchmarkCollector:
    """Collects timing + system metrics and persists them to `benchmarks/`."""

    def __init__(self, benchmarks_dir: Path | str = "benchmarks") -> None:
        """Initialize the collector with an output directory.

        Args:
            benchmarks_dir: Directory where benchmark JSON files will be written.
        """
        self._bench_dir = Path(benchmarks_dir)
        self._bench_dir.mkdir(exist_ok=True)
        self._start_time: float | None = None
        self._model_load_time: float | None = None  # placeholder, may be set externally
        # Live sampling state
        self._samples: list[dict[str, Any]] = []  # raw snapshots
        self._sampling_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._sample_interval: float = 0.5  # seconds

    # ---------------------------------------------------------------------
    # Timing helpers
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Mark the start of wall-clock measurement and begin live sampling."""
        self._start_time = time.perf_counter()
        # initialise sampler
        self._stop_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sample_loop, daemon=True)
        logger.debug(
            "Starting live GPU/RAM sampling (interval %.2fs)", self._sample_interval
        )
        self._sampling_thread.start()

    def stop(self) -> float:
        """Return elapsed time since :meth:`start` in seconds.

        Raises:
            RuntimeError: If called before :meth:`start` was invoked.
        """
        if self._start_time is None:
            raise RuntimeError("BenchmarkCollector.stop() called before start().")
        return time.perf_counter() - self._start_time

    def stop_sampling(self) -> None:
        """Signal the sampler thread to stop and wait for it."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join()
        logger.debug(
            "Stopped live sampling after %.2fs; collected %d samples",
            (time.perf_counter() - self._start_time) if self._start_time else 0,
            len(self._samples),
        )

    def set_model_load_time(self, seconds: float) -> None:
        """Record the model loading time in seconds.

        Args:
            seconds: Duration of model initialization.
        """
        self._model_load_time = seconds

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def _sample_loop(self) -> None:
        """Background thread that samples system/GPU stats periodically."""
        while self._stop_event is not None and not self._stop_event.is_set():
            snapshot = {
                "elapsed_s": (
                    time.perf_counter() - self._start_time if self._start_time else 0
                ),
                "system": self._collect_system_metrics(),
                "gpu": self._collect_gpu_metrics(),
            }
            self._samples.append(snapshot)
            # wait for interval
            self._stop_event.wait(self._sample_interval)

    def _avg_system_metrics(self) -> dict[str, Any]:
        """Average numeric system metrics across samples.

        Returns:
            A dictionary of averaged system metrics rounded to two decimals.
        """
        if not self._samples:
            return {}
        acc: dict[str, float] = {}
        count = len(self._samples)
        for snap in self._samples:
            for k, v in snap["system"].items():
                if isinstance(v, (int, float)):
                    acc[k] = acc.get(k, 0.0) + float(v)
        return {k: round(v / count, 2) for k, v in acc.items()}

    def _average_metrics(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return average system and GPU metrics over all collected samples.

        Returns:
            A tuple containing averaged system metrics and averaged GPU metrics
            (or None if GPU metrics are unavailable).
        """
        if not self._samples:
            return self._collect_system_metrics(), self._collect_gpu_metrics()

        # Initialize accumulators
        sys_acc: dict[str, float] = {}
        gpu_acc: dict[str, float] | None = None
        count = len(self._samples)
        for snap in self._samples:
            sys_metrics = snap["system"]
            for k, v in sys_metrics.items():
                if isinstance(v, (int, float)):
                    sys_acc[k] = sys_acc.get(k, 0.0) + float(v)
            gpu_metrics = snap.get("gpu")
            if gpu_metrics is not None:
                if gpu_acc is None:
                    gpu_acc = {}
                for k, v in gpu_metrics.items():
                    if isinstance(v, (int, float)):
                        gpu_acc[k] = gpu_acc.get(k, 0.0) + float(v)
        # average
        sys_avg = {k: round(v / count, 2) for k, v in sys_acc.items()}
        gpu_avg = None
        if gpu_acc is not None:
            gpu_avg = {k: round(v / count, 2) for k, v in gpu_acc.items()}
        return sys_avg, gpu_avg

    def _gpu_vram_stats(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        """Return (min_non_zero, max, median) for vram_used_mb across samples."""
        min_val: float | None = None
        max_val: float | None = None
        values: list[float] = []
        for snap in self._samples:
            gpu_metrics = snap.get("gpu") or {}
            val = gpu_metrics.get("vram_used_mb")
            if val is None or val == 0:
                continue
            values.append(val)
            if min_val is None or val < min_val:
                min_val = val
            if max_val is None or val > max_val:
                max_val = val
        med_val = None
        if values:
            med_val = median(values)
        return (
            round(min_val, 2) if min_val is not None else None,
            round(max_val, 2) if max_val is not None else None,
            round(med_val, 2) if med_val is not None else None,
        )

    def collect(
        self,
        *,
        audio_path: str,
        task: str,
        config: dict[str, Any] | None,
        runtime_seconds: float | None,
        total_time: float | None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Gather metrics and write to a timestamped JSON file.

        Stops live sampling and includes all collected samples.

        Returns:
            Path: The path to the JSON file that was written.
        """
        # Ensure sampling thread stopped
        self.stop_sampling()

        # Compute averages
        avg_sys, avg_gpu = self._average_metrics()
        vram_min, vram_max, vram_median = self._gpu_vram_stats()
        vram_avg = avg_gpu.get("vram_used_mb") if avg_gpu else None
        sample_count = len(self._samples) if self._samples else 0

        # Memory stats
        ram_total = self._collect_system_metrics().get("ram_total_mb")
        ram_avg = avg_sys.get("ram_used_mb") if avg_sys else None

        gpu_dict = None
        if self._collect_gpu_metrics() is not None:
            gpu_dict = {
                "name": self._collect_gpu_metrics().get("name"),
                "total_vram_mb": self._collect_gpu_metrics().get("total_vram_mb"),
                "gpu_vram_min_mb": vram_min,
                "gpu_vram_max_mb": vram_max,
                "gpu_vram_median_mb": vram_median,
                "gpu_vram_avarage_mb": vram_avg,
                "sample_count": sample_count,
            }

        memory_dict = None
        if ram_total is not None:
            memory_dict = {
                "ram_total_mb": ram_total,
                "ram_avarage_mb": ram_avg,
                "sample_count": sample_count,
            }

        result = BenchmarkResult(
            timestamp=datetime.utcnow().isoformat(),
            model=(config or {}).get("model"),
            device=(config or {}).get("device"),
            batch_size=(config or {}).get("batch_size"),
            runtime_seconds=runtime_seconds,
            total_wall_time_seconds=total_time,
            model_load_seconds=self._model_load_time,
            system={
                k: v
                for k, v in self._collect_system_metrics().items()
                if k not in {"ram_total_mb", "ram_used_mb"}
            },  # final snapshot
            gpu=gpu_dict,
            memory=memory_dict,
            extra=extra,
        )

        # Generate filename using the shared FilenameGenerator, then prepend
        # "benchmark_".
        strategy = StandardFilenameStrategy()
        generator = FilenameGenerator(strategy=strategy)
        try:
            task_enum = TaskType(task)
        except ValueError:
            task_enum = TaskType.TRANSCRIBE  # fallback
        base_name = generator.create_filename(
            audio_path=audio_path,
            task=task_enum,
            extension="json",
        )
        filename = f"benchmark_{base_name}"
        path = self._bench_dir / filename

        # Serialize the Pydantic model in a way that works for both Pydantic v1 and v2.
        try:
            # Pydantic v2 preferred API
            json_str: str = result.model_dump_json(indent=2)  # type: ignore[attr-defined]
        except AttributeError:  # Fallback to Pydantic v1
            json_str = result.json(indent=2)  # type: ignore[call-arg]

        path.write_text(json_str, encoding="utf-8")  # type: ignore[arg-type]
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_system_metrics() -> dict[str, Any]:
        """Collect basic system-wide memory info (RAM).

        Returns:
            A dictionary with OS, Python, and memory statistics.
        """
        data: dict[str, Any] = {
            "os": platform.platform(),
            "python_version": platform.python_version(),
        }
        if torch is not None:
            data["torch_version"] = torch.__version__  # type: ignore[attr-defined]

        if psutil is not None:
            vm = psutil.virtual_memory()  # type: ignore[attr-defined]
            data.update({
                "ram_total_mb": round(vm.total / 1024**2, 2),
                "ram_used_mb": round(vm.used / 1024**2, 2),
            })
        return data

    @staticmethod
    def _collect_gpu_metrics() -> dict[str, Any] | None:
        """Return AMD GPU statistics if `pyamdgpuinfo` is installed."""
        # Try CUDA first
        if torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
            try:
                device_idx = torch.cuda.current_device()  # type: ignore[attr-defined]
                return {
                    "name": torch.cuda.get_device_name(device_idx),  # type: ignore[attr-defined]
                    "total_vram_mb": round(
                        torch.cuda.get_device_properties(device_idx).total_memory
                        / 1024**2,
                        2,
                    ),  # type: ignore[attr-defined]
                    "vram_used_mb": round(
                        torch.cuda.memory_allocated(device_idx) / 1024**2, 2
                    ),  # type: ignore[attr-defined]
                }
            except Exception:  # pragma: no cover
                pass

        # Fallback to AMD via pyamdgpuinfo
        if pyamdgpuinfo is None:
            return None
        try:
            cards = pyamdgpuinfo.get_cards()  # type: ignore[attr-defined]
            if not cards:
                return None
            card = cards[0]
            return {
                "name": card.get_name(),
                "total_vram_mb": round(card.get_vram_total() / 1024**2, 2),
                "vram_used_mb": round(card.get_vram_used() / 1024**2, 2),
                "temperature_c": card.get_temp(),
                "power_w": card.get_power_avg(),
            }
        except Exception:  # pragma: no cover – best-effort
            return None
