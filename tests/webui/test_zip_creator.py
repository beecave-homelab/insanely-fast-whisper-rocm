"""Tests for ZIP archive creation helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from insanely_fast_whisper_rocm.webui.zip_creator import (
    BatchZipBuilder,
    ZipConfiguration,
    create_batch_zip,
)


def test_add_batch_files_organized_by_format(tmp_path: Path) -> None:
    """Write files under per-format folders when organize_by_format is enabled."""
    cfg = ZipConfiguration(temp_dir=str(tmp_path), organize_by_format=True)

    raw = {"text": "hello", "chunks": [{"text": "hello", "timestamp": [0.0, 1.0]}]}

    builder = BatchZipBuilder(cfg)
    with builder.create(batch_id="b1", filename="out.zip") as b:
        b.add_batch_files({"/tmp/a.mp3": raw}, formats=["txt", "json"])
        zip_path, stats = b.build()

    assert stats.files_added == 3  # txt + json + summary

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "txt/a.txt" in names
    assert "json/a.json" in names
    assert "batch_summary.json" in names


def test_add_merged_files_creates_expected_paths(tmp_path: Path) -> None:
    """Create merged outputs under merged/ with proper extensions."""
    cfg = ZipConfiguration(temp_dir=str(tmp_path), include_summary=False)

    raw = {"text": "hello"}
    builder = BatchZipBuilder(cfg)
    with builder.create(batch_id="b1", filename="out.zip") as b:
        b.add_merged_files(
            {"/tmp/a.mp3": raw}, formats=["txt", "json"], merged_filename="m"
        )
        zip_path, stats = b.build()

    assert stats.files_added == 2

    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "merged/m.txt" in zf.namelist()
        assert "merged/m.json" in zf.namelist()


def test_get_base_filename_sanitizes_and_falls_back(tmp_path: Path) -> None:
    """Sanitize unsafe characters and provide a fallback for empty names."""
    cfg = ZipConfiguration(temp_dir=str(tmp_path), include_summary=False)
    builder = BatchZipBuilder(cfg)

    assert builder._get_base_filename('/tmp/abc<>:"|?*.mp3') == "abc_______"
    assert builder._get_base_filename("/tmp/....mp3") == "file"


def test_add_summary_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignore summary write errors and record them in stats."""
    cfg = ZipConfiguration(temp_dir=str(tmp_path), include_summary=True)
    builder = BatchZipBuilder(cfg)

    monkeypatch.setattr(
        builder, "_generate_summary", Mock(side_effect=TypeError("bad"))
    )

    with builder.create(batch_id="b1", filename="out.zip") as b:
        b.add_summary(include_stats=True)
        zip_path, stats = b.build()

    assert Path(zip_path).exists()
    assert any("Failed to add summary" in e for e in stats.errors)


def test_create_batch_zip_convenience_can_include_merged(tmp_path: Path) -> None:
    """Create a zip with individual + merged content via helper."""
    cfg = ZipConfiguration(temp_dir=str(tmp_path), include_summary=False)

    file_results = {"/tmp/a.mp3": {"text": "hello"}, "/tmp/b.mp3": {"text": "world"}}
    zip_path, stats = create_batch_zip(
        file_results,
        formats=["txt", "json"],
        batch_id="b1",
        include_merged=True,
        config=cfg,
    )

    assert stats.files_added == 6  # 2 files * 2 formats + 2 merged

    with zipfile.ZipFile(zip_path, "r") as zf:
        assert "merged/batch_merged_b1.txt" in zf.namelist()
        merged_json = json.loads(zf.read("merged/batch_merged_b1.json").decode("utf-8"))

    assert merged_json["batch_info"]["total_files"] == 2
