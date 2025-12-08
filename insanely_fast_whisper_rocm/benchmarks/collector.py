"""Benchmark collection utilities for CLI instrumentation.

The collector persists a JSON document for each CLI run that enables
post-mortem analysis. Besides core runtime details, the record now captures
every CLI flag value and optional GPU utilisation statistics gathered via
``pyamdgpuinfo`` when the library and compatible hardware are available.

Benchmark outputs reside in ``benchmarks/`` by default and use a safe slug
derived from the audio filename plus a timestamp (in the configured
application timezone) for uniqueness.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from insanely_fast_whisper_rocm.utils.constants import APP_TIMEZONE

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pyamdgpuinfo  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyamdgpuinfo = None  # type: ignore


class BenchmarkCollector:
    """Collect and persist benchmarking metadata for CLI runs.

    Args:
        output_dir: Directory where benchmark JSON files will be written.
            Defaults to ``"benchmarks"`` relative to the current working
            directory.
    """

    _SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, output_dir: Path | str = "benchmarks") -> None:
        """Initialize the collector with an output directory.

        Args:
            output_dir: Target directory for benchmark JSON files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        *,
        audio_path: str,
        task: str,
        config: dict[str, Any] | None,
        runtime_seconds: float | None,
        total_time: float,
        extra: dict[str, str] | None = None,
        gpu_stats: dict[str, Any] | None = None,
        format_quality: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a benchmark record to disk and return its path.

        Args:
            audio_path: Path to the processed audio input.
            task: Executed task name (``"transcribe"`` or ``"translate"``).
            config: Configuration dictionary used for the run.
            runtime_seconds: Model execution time as reported by the backend.
            total_time: End-to-end wall clock time.
            extra: Optional dictionary of additional metadata entries.
            gpu_stats: Aggregated GPU utilisation metrics, if available.
            format_quality: Optional mapping of format-specific quality metrics
                (e.g., {"srt": {"score": 0.95, "details": {...}}}).

        Returns:
            Path: Location of the written benchmark JSON file.
        """
        # Resolve target timezone from centralized configuration, fallback to UTC
        try:
            target_tz = ZoneInfo(APP_TIMEZONE)
        except ZoneInfoNotFoundError:
            target_tz = ZoneInfo("UTC")

        logger.debug(
            "BenchmarkCollector.collect: audio=%s, task=%s, runtime=%.2fs, "
            "total_time=%.2fs, has_gpu_stats=%s",
            audio_path,
            task,
            runtime_seconds or 0.0,
            total_time,
            gpu_stats is not None,
        )

        record = {
            "audio_path": audio_path,
            "task": task,
            "config": config or {},
            "runtime_seconds": runtime_seconds,
            "total_time_seconds": total_time,
            "extra": extra or {},
            "gpu_stats": gpu_stats or {},
            "format_quality": format_quality or {},
            # Keep ISO8601; timezone-aware using configured timezone
            "recorded_at": datetime.now(target_tz).isoformat(),
        }
        logger.debug(
            "Benchmark record: format_quality_keys=%s, extra_keys=%s",
            list((format_quality or {}).keys()),
            list((extra or {}).keys()),
        )

        slug = self._slugify(Path(audio_path).stem or "benchmark")
        # Use UTC for filenames to ensure consistent, timezone-agnostic ordering
        timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{slug}_{task}_{timestamp}.json"
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.debug("Benchmark written to: %s", output_path)
        return output_path

    @classmethod
    def _slugify(cls, value: str) -> str:
        """Normalize ``value`` for safe filesystem usage.

        Args:
            value: Raw identifier to sanitize.

        Returns:
            str: A filesystem-safe slug representation of ``value``.
        """
        sanitized = cls._SAFE_NAME_PATTERN.sub("-", value).strip("-._")
        return sanitized or "benchmark"


class GpuUtilSampler:
    """Sample GPU utilisation metrics using ``pyamdgpuinfo``.

    The sampler polls the primary GPU at a fixed interval to build an average
    utilisation profile. When ``pyamdgpuinfo`` or compatible hardware is not
    available, the sampler becomes a no-op and returns ``None``.
    """

    def __init__(self, interval: float = 0.5) -> None:
        """Initialise the sampler.

        Args:
            interval: Sampling interval in seconds.
        """
        self._interval = interval
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._gpu = None

    def start(self) -> bool:
        """Begin sampling in a background thread.

        Returns:
            ``True`` when sampling started successfully, ``False`` otherwise.
        """
        if pyamdgpuinfo is None:  # pragma: no cover - optional dependency
            return False
        try:  # pragma: no cover - hardware specific
            self._gpu = pyamdgpuinfo.get_gpu(0)
        except Exception:
            self._gpu = None
            return False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop sampling and wait for the background thread to finish."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
        self._stop_event = None
        self._thread = None

    def summary(self) -> dict[str, Any] | None:
        """Return aggregated utilisation metrics.

        Returns:
            A dictionary containing average and peak GPU utilisation and VRAM
            usage. ``None`` is returned when no samples were collected.
        """
        if not self._samples:
            return None
        loads = [sample[0] for sample in self._samples]
        vram_bytes = [sample[1] for sample in self._samples]
        sample_count = len(self._samples)
        return {
            "provider": "pyamdgpuinfo",
            "sample_interval_seconds": self._interval,
            "sample_count": sample_count,
            "avg_gpu_load_percent": round((sum(loads) / sample_count) * 100, 2),
            "max_gpu_load_percent": round(max(loads) * 100, 2),
            "avg_vram_mb": round((sum(vram_bytes) / sample_count) / 1024**2, 2),
            "max_vram_mb": round(max(vram_bytes) / 1024**2, 2),
        }

    def _run_loop(self) -> None:
        """Continuously capture GPU utilisation until stopped."""
        if self._gpu is None or self._stop_event is None:
            return
        while not self._stop_event.is_set():  # pragma: no cover - hardware specific
            try:
                load = float(self._gpu.query_load())
                vram = float(self._gpu.query_vram_usage())
            except Exception:
                break
            self._samples.append((load, vram))
            self._stop_event.wait(self._interval)
