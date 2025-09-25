"""Benchmark collection utilities for CLI instrumentation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    ) -> Path:
        """Persist a benchmark record to disk and return its path.

        Args:
            audio_path: Path to the processed audio input.
            task: Executed task name (``"transcribe"`` or ``"translate"``).
            config: Configuration dictionary used for the run.
            runtime_seconds: Model execution time as reported by the backend.
            total_time: End-to-end wall clock time.
            extra: Optional dictionary of additional metadata entries.

        Returns:
            Path: Location of the written benchmark JSON file.
        """
        record = {
            "audio_path": audio_path,
            "task": task,
            "config": config or {},
            "runtime_seconds": runtime_seconds,
            "total_time_seconds": total_time,
            "extra": extra or {},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        slug = self._slugify(Path(audio_path).stem or "benchmark")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{slug}_{task}_{timestamp}.json"
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
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
