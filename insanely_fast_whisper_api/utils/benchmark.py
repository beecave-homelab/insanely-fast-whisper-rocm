"""Benchmark utilities for collecting and persisting performance metrics.

This module is *optional* – it only incurs extra dependencies (`psutil`,
`pyamdgpuinfo`, `pydantic`) and minor overhead when the CLI is executed with
`--benchmark`. When the flag is absent, importing this module is avoided to
keep startup time minimal.
"""

from __future__ import annotations

import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

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
    from pydantic import BaseModel  # type: ignore
except ImportError:  # pragma: no cover
    # The core package already depends on pydantic; but guard just in case
    BaseModel = object  # type: ignore

__all__ = [
    "BenchmarkResult",
    "BenchmarkCollector",
]


class BenchmarkResult(BaseModel):  # type: ignore[misc]
    """Schema for a single benchmark result entry."""

    timestamp: str  # ISO8601
    model: Optional[str]
    device: Optional[str]
    runtime_seconds: Optional[float]
    total_wall_time_seconds: Optional[float]
    model_load_seconds: Optional[float] = None

    # System metrics
    system: Dict[str, Any]
    gpu: Dict[str, Any] | None = None

    extra: Dict[str, Any] | None = None

    class Config:
        frozen = True  # make it hashable / safe


class BenchmarkCollector:
    """Collects timing + system metrics and persists them to `benchmarks/`."""

    def __init__(self, benchmarks_dir: Path | str = "benchmarks") -> None:
        self._bench_dir = Path(benchmarks_dir)
        self._bench_dir.mkdir(exist_ok=True)
        self._start_time: Optional[float] = None
        self._model_load_time: Optional[float] = (
            None  # placeholder, may be set externally
        )

    # ---------------------------------------------------------------------
    # Timing helpers
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Mark the start of wall-clock measurement."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed time since :meth:`start` in seconds."""
        if self._start_time is None:
            raise RuntimeError("BenchmarkCollector.stop() called before start().")
        return time.perf_counter() - self._start_time

    def set_model_load_time(self, seconds: float) -> None:
        self._model_load_time = seconds

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def collect(
        self,
        *,
        audio_path: str,
        task: str,
        config: Dict[str, Any] | None,
        runtime_seconds: Optional[float],
        total_time: Optional[float],
        extra: Dict[str, Any] | None = None,
    ) -> Path:
        """Gather metrics & write to a timestamped JSON file.

        Returns the path to the JSON file that was written.
        """
        result = BenchmarkResult(
            timestamp=datetime.utcnow().isoformat(),
            model=(config or {}).get("model"),
            device=(config or {}).get("device"),
            runtime_seconds=runtime_seconds,
            total_wall_time_seconds=total_time,
            model_load_seconds=self._model_load_time,
            system=self._collect_system_metrics(),
            gpu=self._collect_gpu_metrics(),
            extra=extra,
        )

        # Generate filename using the shared FilenameGenerator, then prepend "benchmark_".
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
    def _collect_system_metrics() -> Dict[str, Any]:
        """Collect basic system-wide memory info (RAM)."""
        data: Dict[str, Any] = {
            "os": platform.platform(),
            "python_version": platform.python_version(),
        }
        if torch is not None:
            data["torch_version"] = torch.__version__  # type: ignore[attr-defined]

        if psutil is not None:
            vm = psutil.virtual_memory()  # type: ignore[attr-defined]
            data.update(
                {
                    "ram_total_mb": round(vm.total / 1024**2, 2),
                    "ram_used_mb": round(vm.used / 1024**2, 2),
                }
            )
        return data

    @staticmethod
    def _collect_gpu_metrics() -> Dict[str, Any] | None:
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
