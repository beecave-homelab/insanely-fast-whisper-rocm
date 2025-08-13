"""CLI command implementations.

Each command is a thin wrapper around [_run_task](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_api/cli/commands.py:380:0-632:19), receiving shared CLI
options via the [audio_options](cci:1://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_api/cli/common_options.py:21:0-150:15) decorator (see [cli/common_options.py](cci:7://file:///home/elvee/Local-AI/insanely-fast-whisper-rocm/insanely_fast_whisper_api/cli/common_options.py:0:0-0:0)).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import click
from click.core import ParameterSource

from insanely_fast_whisper_api.audio.processing import extract_audio_from_video
from insanely_fast_whisper_api.cli.common_options import audio_options
from insanely_fast_whisper_api.cli.facade import cli_facade
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.core.formatters import FORMATTERS
from insanely_fast_whisper_api.utils import constants
from insanely_fast_whisper_api.utils.file_utils import cleanup_temp_files
from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# High-level command wrappers                                                 #
# --------------------------------------------------------------------------- #


@click.command(short_help="Transcribe an audio file")
@audio_options
def transcribe(audio_file: Path, **kwargs) -> None:
    """Transcribe *audio_file* using Whisper models."""

    # Was --export-format explicitly supplied?
    ctx = click.get_current_context()
    kwargs["export_format_explicit"] = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(task="transcribe", audio_file=audio_file, **kwargs)


@click.command(short_help="Translate an audio file to English")
@audio_options
def translate(audio_file: Path, **kwargs) -> None:
    """Translate *audio_file* to English using Whisper models."""

    ctx = click.get_current_context()
    kwargs["export_format_explicit"] = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(task="translate", audio_file=audio_file, **kwargs)


# --------------------------------------------------------------------------- #
# Core execution logic                                                        #
# --------------------------------------------------------------------------- #


def _run_task(*, task: str, audio_file: Path, **kwargs) -> None:  # noqa: C901
    """
    Execute *task* (“transcribe” or “translate”) on *audio_file*.

    All CLI flags arrive in **kwargs.
    """
    start_time = time.time()

    # ------------------------------------------------------------------ #
    # Extract and normalise arguments                                    #
    # ------------------------------------------------------------------ #
    model: str = kwargs.pop("model")
    backend_type: str = kwargs.pop("backend")
    device: str = kwargs.pop("device")
    dtype: str = kwargs.pop("dtype")
    batch_size: int = kwargs.pop("batch_size")
    chunk_length: int = kwargs.pop("chunk_length")
    language: str = kwargs.pop("language")
    output: Optional[Path] = kwargs.pop("output", None)
    timestamp_type: str = kwargs.pop("timestamp_type")
    # Stable-ts options
    stabilize: bool = kwargs.pop("stabilize")
    demucs: bool = kwargs.pop("demucs")
    vad: bool = kwargs.pop("vad")
    vad_threshold: float = kwargs.pop("vad_threshold")

    debug: bool = kwargs.pop("debug", False)
    no_timestamps: bool = kwargs.pop("no_timestamps", False)
    export_format: str = kwargs.pop("export_format", "json")
    benchmark: bool = kwargs.pop("benchmark", False)
    benchmark_extra: tuple[str, ...] = kwargs.pop("benchmark_extra", ())
    export_format_explicit: bool = kwargs.pop("export_format_explicit", False)

    # Legacy flags --export-json/--export-srt/--export-txt/--export-all.
    # They’re parsed in common_options and arrive here as boolean kwargs.
    # Map them back to `export_format`.
    if kwargs.pop("export_json", False):
        export_format, export_format_explicit = "json", True
    if kwargs.pop("export_srt", False):
        export_format, export_format_explicit = "srt", True
    if kwargs.pop("export_txt", False):
        export_format, export_format_explicit = "txt", True
    if kwargs.pop("export_all", False):
        export_format, export_format_explicit = "all", True

    # Pop any stray kwargs we don’t explicitly handle
    kwargs.clear()

    # ------------------------------------------------------------------ #
    # Pre-processing (video handling, language auto-detect, etc.)         #
    # ------------------------------------------------------------------ #
    processed_language = language or None
    temp_files: list[Path] = []

    # If a video file was supplied, extract its audio first
    if audio_file.suffix.lower() in constants.SUPPORTED_VIDEO_FORMATS:
        audio_file = extract_audio_from_video(video_path=audio_file)
        temp_files.append(audio_file)

    # ------------------------------------------------------------------ #
    # Execute the ASR backend via the facade                              #
    # ------------------------------------------------------------------ #
    try:
        return_timestamps_value = False
        if not no_timestamps:
            return_timestamps_value = "word" if timestamp_type == "word" else True

        result = cli_facade.process_audio(
            audio_file_path=audio_file,
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
            language=processed_language,
            task=task,
            return_timestamps_value=return_timestamps_value,
            backend=backend_type,
        )

        # Optional stable-ts post-processing
        if stabilize:
            try:
                from insanely_fast_whisper_api.core.integrations import (
                    stabilize_timestamps,
                )

                # Ensure the result contains the audio path for stable-ts
                result.setdefault("audio_file_path", str(audio_file))

                result = stabilize_timestamps(
                    result,
                    demucs=demucs,
                    vad=vad,
                    vad_threshold=vad_threshold,
                )
            except Exception as exc:  # pragma: no cover
                click.secho(f"⚠️  stable-ts post-processing failed: {exc}", fg="yellow")

        # INFO-level summary (lazy logging)
        logger.info(
            "Segments: %s | Stabilized: %s (%s)",
            result.get("segments_count"),
            bool(result.get("stabilized")),
            result.get("stabilization_path", "n/a"),
        )

        total_time = time.time() - start_time

        # ------------------------------------------------------------------ #
        # Export & benchmark handling                                         #
        # ------------------------------------------------------------------ #
        _handle_output_and_benchmarks(
            task=task,
            audio_file=audio_file,
            result=result,
            total_time=total_time,
            output=output,
            export_format=export_format,
            export_format_explicit=export_format_explicit,
            benchmark_enabled=benchmark,
            benchmark_extra=benchmark_extra,
            temp_files=temp_files,
        )

    # ------------------------------------------------------------------ #
    # Error handling                                                      #
    # ------------------------------------------------------------------ #
    except DeviceNotFoundError as exc:
        click.secho(f"\n❌ Device error: {exc}", fg="red", err=True)
        click.secho(
            "💡 Try --device cpu or verify your CUDA/MPS setup", fg="yellow", err=True
        )
        if debug:
            logger.exception("Device error details")
        sys.exit(1)

    except TranscriptionError as exc:
        click.secho(f"\n❌ {task.capitalize()} error: {exc}", fg="red", err=True)
        if debug:
            logger.exception("%s error details", task)
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        click.secho(f"\n❌ Unexpected error: {exc}", fg="red", err=True)
        if debug:
            logger.exception("Unexpected error during %s", task)
        else:
            click.secho(
                "💡 Re-run with --debug for more details", fg="yellow", err=True
            )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Helper: output files + benchmark                                            #
# --------------------------------------------------------------------------- #


def _handle_output_and_benchmarks(
    *,
    task: str,
    audio_file: Path,
    result: dict,
    total_time: float,
    output: Optional[Path],
    export_format: str,
    export_format_explicit: bool,
    benchmark_enabled: bool,
    benchmark_extra: tuple[str, ...],
    temp_files: list[Path],
) -> None:
    """Handle file export and benchmark writing."""

    # Decide which formats to export
    if export_format == "all":
        formats_to_export = ("json", "txt", "srt")
    else:
        formats_to_export = (export_format,)

    # Build a “detailed result” once for all formatters
    detailed_result = {
        task: result["text"],
        "text": result["text"],  # TxtFormatter uses this
        # Prefer the stable-ts key name
        "segments": result.get("segments") or result.get("chunks", []),
        "metadata": {
            "audio_file": str(audio_file.resolve()),
            "total_time_seconds": round(total_time, 2),
            "processing_time_seconds": result.get("runtime_seconds"),
            "config_used": result.get("config_used", {}),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # ------------------------------------------------------------------ #
    # Export each requested format                                       #
    # ------------------------------------------------------------------ #
    for fmt in formats_to_export:
        formatter = FORMATTERS[fmt]
        content = formatter.format(detailed_result)
        ext = formatter.get_file_extension()

        if output and export_format_explicit and fmt != "all":
            output_path = (
                output if fmt == export_format else output.with_suffix(f".{ext}")
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: transcripts/ (json) or transcripts-{fmt}/
            if fmt == "json":
                out_dir = Path(constants.DEFAULT_TRANSCRIPTS_DIR)
            else:
                out_dir = Path(f"transcripts-{fmt}")
            out_dir.mkdir(exist_ok=True)

            filename_gen = FilenameGenerator(strategy=StandardFilenameStrategy())
            output_path = out_dir / filename_gen.create_filename(
                audio_path=str(audio_file),
                task=TaskType(task),
                extension=ext,
            )

        try:
            output_path.write_text(content, encoding="utf-8")
            click.secho(f"💾 Saved {fmt.upper()} to: {output_path}", fg="green")
        except OSError as exc:
            click.secho(f"❌ Failed to save {fmt.upper()}: {exc}", fg="red", err=True)

    # ------------------------------------------------------------------ #
    # Benchmark (optional)                                               #
    # ------------------------------------------------------------------ #
    if benchmark_enabled:
        from insanely_fast_whisper_api.benchmarks.collector import BenchmarkCollector

        collector = BenchmarkCollector()
        extra_dict: Dict[str, str] | None = None
        if benchmark_extra:
            extra_dict = dict(item.split("=", 1) for item in benchmark_extra)

        benchmark_path = collector.collect(
            audio_path=str(audio_file),
            task=task,
            config=result.get("config_used"),
            runtime_seconds=result.get("runtime_seconds"),
            total_time=total_time,
            extra=extra_dict,
        )
        click.secho(f"📈 Benchmark saved to: {benchmark_path}", fg="green")

    # ------------------------------------------------------------------ #
    # Cleanup                                                            #
    # ------------------------------------------------------------------ #
    if temp_files:
        cleanup_temp_files(temp_files)
