"""CLI command implementations.

Each command is a thin wrapper around the internal ``_run_task`` function and
receives shared CLI options via the ``audio_options`` decorator. See
``cli/common_options.py`` for the shared option definitions.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from insanely_fast_whisper_rocm.audio.processing import extract_audio_from_video
from insanely_fast_whisper_rocm.cli.common_options import audio_options
from insanely_fast_whisper_rocm.cli.facade import cli_facade
from insanely_fast_whisper_rocm.cli.progress_tqdm import TqdmProgressReporter
from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.errors import (
    DeviceNotFoundError,
    TranscriptionCancelledError,
    TranscriptionError,
)
from insanely_fast_whisper_rocm.core.formatters import (
    FORMATTERS,
    build_quality_segments,
)
from insanely_fast_whisper_rocm.core.progress import ProgressCallback
from insanely_fast_whisper_rocm.utils import constants
from insanely_fast_whisper_rocm.utils.file_utils import cleanup_temp_files
from insanely_fast_whisper_rocm.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)
from insanely_fast_whisper_rocm.utils.srt_quality import compute_srt_quality

try:
    from insanely_fast_whisper_rocm.core.integrations import stabilize_timestamps
except ModuleNotFoundError:  # pragma: no cover

    def stabilize_timestamps(  # type: ignore[no-redef]
        result: dict[str, Any],
        *,
        demucs: bool = False,
        vad: bool = False,
        vad_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Raise a helpful error when stable-ts integration is unavailable.

        Raises:
            RuntimeError: Always, indicating the optional stable-ts dependency
                is missing from the current installation.
        """
        raise RuntimeError(
            "stable-ts integration is not installed; reinstall with the extra"
            " dependencies to enable --stabilize support."
        )


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_output_fds() -> Generator[None, None, None]:
    """Temporarily silence stdout/stderr at the file-descriptor level.

    This suppresses outputs from C/C++ libraries and progress bars (e.g.,
    MIOpen/Demucs/tqdm) that bypass Python's logging and write directly to
    the underlying file descriptors.

    Yields:
        None: Control to the with-block where output remains silenced.
    """
    # Save original fds
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)  # stdout -> /dev/null
        os.dup2(devnull_fd, 2)  # stderr -> /dev/null
        yield
    finally:
        # Restore
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


# --------------------------------------------------------------------------- #
# High-level command wrappers                                                 #
# --------------------------------------------------------------------------- #


@click.command(short_help="Transcribe an audio file")
@audio_options
def transcribe(audio_file: Path, **kwargs: dict) -> None:
    """Transcribe *audio_file* using Whisper models."""
    # Was --export-format explicitly supplied?
    ctx = click.get_current_context()
    kwargs["export_format_explicit"] = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(task="transcribe", audio_file=audio_file, **kwargs)


@click.command(short_help="Translate an audio file to English")
@audio_options
def translate(audio_file: Path, **kwargs: dict) -> None:
    """Translate *audio_file* to English using Whisper models."""
    ctx = click.get_current_context()
    kwargs["export_format_explicit"] = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(task="translate", audio_file=audio_file, **kwargs)


# --------------------------------------------------------------------------- #
# Core execution logic                                                        #
# --------------------------------------------------------------------------- #


def _is_stabilization_corrupt(segments: list[dict]) -> bool:
    """Check if the stabilized segments appear to be corrupt.

    Returns:
        bool: True if the segments are likely corrupt, False otherwise.
    """
    if not segments or len(segments) < 2:
        return False

    # Heuristic: If > 50% of segments have identical timestamps, it's corrupt.
    first_timestamp = (segments[0].get("start"), segments[0].get("end"))
    identical_count = sum(
        1 for seg in segments if (seg.get("start"), seg.get("end")) == first_timestamp
    )

    return (identical_count / len(segments)) > 0.5


def _run_task(*, task: str, audio_file: Path, **kwargs: dict) -> None:  # noqa: C901
    """Execute *task* ("transcribe" or "translate") on *audio_file*.

    All CLI flags arrive in **kwargs.
    """
    start_time = time.time()
    logger.debug(
        "Starting task: %s, audio=%s",
        task,
        audio_file,
    )

    # ------------------------------------------------------------------ #
    # Extract and normalise arguments                                    #
    # ------------------------------------------------------------------ #
    model: str = kwargs.pop("model")
    device: str = kwargs.pop("device")
    dtype: str = kwargs.pop("dtype")
    batch_size: int = kwargs.pop("batch_size")
    progress_group_size: int = kwargs.pop("progress_group_size")
    chunk_length: int = kwargs.pop("chunk_length")
    language: str = kwargs.pop("language")
    output: Path | None = kwargs.pop("output", None)
    timestamp_type: str = kwargs.pop("timestamp_type")
    # Stable-ts options
    stabilize: bool = kwargs.pop("stabilize")
    demucs: bool = kwargs.pop("demucs")
    vad: bool = kwargs.pop("vad")
    vad_threshold: float = kwargs.pop("vad_threshold")

    debug: bool = kwargs.pop("debug", False)
    quiet: bool = kwargs.pop("quiet", False)
    progress_enabled: bool = kwargs.pop("progress", True)
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

    benchmark_flags: dict[str, Any] | None = None
    benchmark_gpu_stats: dict[str, Any] | None = None
    gpu_sampler: object | None = None

    # ------------------------------------------------------------------ #
    # Pre-processing (video handling, language auto-detect, etc.)         #
    # ------------------------------------------------------------------ #
    processed_language = language or None
    temp_files: list[Path] = []
    reporter = TqdmProgressReporter(enabled=progress_enabled)
    cancellation_token = CancellationToken()

    def _ensure_not_cancelled() -> None:
        if cancellation_token.cancelled:
            raise TranscriptionCancelledError("Transcription cancelled by user")

    previous_signal_handlers: list[tuple[int, Any]] = []

    def _register_signal(sig: int) -> None:
        previous_signal_handlers.append((sig, signal.getsignal(sig)))
        signal.signal(sig, lambda signum, frame: cancellation_token.cancel())

    _register_signal(signal.SIGINT)
    if hasattr(signal, "SIGTERM"):
        _register_signal(signal.SIGTERM)

    if benchmark:
        benchmark_flags = {
            "audio_file": str(audio_file),
            "model": model,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "progress_group_size": progress_group_size,
            "chunk_length": chunk_length,
            "language": processed_language,
            "output": str(output) if output else None,
            "timestamp_type": timestamp_type,
            "stabilize": stabilize,
            "demucs": demucs,
            "vad": vad,
            "vad_threshold": vad_threshold,
            "debug": debug,
            "quiet": quiet,
            "progress": progress_enabled,
            "no_timestamps": no_timestamps,
            "export_format": export_format,
            "export_format_explicit": export_format_explicit,
            "benchmark_extra": list(benchmark_extra),
        }
        from insanely_fast_whisper_rocm.benchmarks.collector import GpuUtilSampler

        sampler = GpuUtilSampler()
        if sampler.start():
            gpu_sampler = sampler

    # Enable DEBUG-level logging when --debug is used
    if debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger("insanely_fast_whisper_rocm").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled via --debug flag")

    # Reduce logging verbosity when --quiet is used
    if quiet and not debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        logging.getLogger("insanely_fast_whisper_rocm").setLevel(logging.ERROR)

    # If a video file was supplied, extract its audio first
    if audio_file.suffix.lower() in constants.SUPPORTED_VIDEO_FORMATS:
        try:
            reporter.on_postprocess_started("extract-audio")
            audio_file = extract_audio_from_video(video_path=audio_file)
            temp_files.append(audio_file)
        finally:
            reporter.on_postprocess_finished("extract-audio")

    _ensure_not_cancelled()

    # ------------------------------------------------------------------ #
    # Execute the ASR backend via the facade                              #
    # ------------------------------------------------------------------ #
    try:
        return_timestamps_value = False
        if not no_timestamps:
            return_timestamps_value = "word" if timestamp_type == "word" else True

        # Configuration details logged by facade at INFO level
        _ensure_not_cancelled()
        result = cli_facade.process_audio(
            audio_file_path=audio_file,
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            chunk_length=chunk_length,
            progress_group_size=progress_group_size,
            language=processed_language,
            task=task,
            return_timestamps_value=return_timestamps_value,
            progress_cb=reporter,
            cancellation_token=cancellation_token,
        )
        _ensure_not_cancelled()
        logger.debug(
            "ASR completed: %d chars, %d chunks, runtime=%.2fs",
            len(result.get("text", "")),
            len(result.get("chunks", [])),
            result.get("runtime_seconds", 0.0),
        )

        # Optional stable-ts post-processing
        if stabilize:
            _ensure_not_cancelled()
            try:
                # Ensure the result contains the audio path for stable-ts
                result.setdefault("audio_file_path", str(audio_file))

                reporter.on_postprocess_started("stable-ts")
                original_result = result
                stabilized_result = None

                _ensure_not_cancelled()

                if quiet:
                    with _suppress_output_fds():
                        stabilized_result = stabilize_timestamps(
                            result,
                            demucs=demucs,
                            vad=vad,
                            vad_threshold=vad_threshold,
                        )
                        _ensure_not_cancelled()
                else:
                    stabilized_result = stabilize_timestamps(
                        result,
                        demucs=demucs,
                        vad=vad,
                        vad_threshold=vad_threshold,
                    )
                    _ensure_not_cancelled()

                if stabilized_result and _is_stabilization_corrupt(
                    stabilized_result.get("segments", [])
                ):
                    if not quiet:
                        click.secho(
                            "⚠️  Stabilization produced corrupted timestamps. "
                            "Falling back to original.",
                            fg="yellow",
                        )
                    result = original_result
                elif stabilized_result:
                    result = stabilized_result
                # If stabilization failed and returned None, result remains
                # original_result
                # Emit granular completion lines instead of a generic message
                if demucs:
                    reporter.on_postprocess_finished("demucs")
                if vad:
                    reporter.on_postprocess_finished(f"vad threshold={vad_threshold}")
            except RuntimeError as exc:
                if not quiet:
                    click.secho(
                        f"⚠️  stable-ts post-processing unavailable: {exc}",
                        fg="yellow",
                    )
            except Exception as exc:  # pragma: no cover
                if not quiet:
                    click.secho(
                        f"⚠️  stable-ts post-processing failed: {exc}", fg="yellow"
                    )

        _ensure_not_cancelled()

        # INFO-level summary (lazy logging) — skip when quiet
        if not quiet:
            logger.info(
                "Segments: %s | Stabilized: %s (%s)",
                result.get("segments_count"),
                bool(result.get("stabilized")),
                result.get("stabilization_path", "n/a"),
            )

        total_time = time.time() - start_time
        logger.debug("Total task execution time: %.2fs", total_time)

        if gpu_sampler is not None:
            gpu_sampler.stop()
            benchmark_gpu_stats = gpu_sampler.summary()
            logger.debug("GPU sampling summary: %s", benchmark_gpu_stats)
            gpu_sampler = None

        _ensure_not_cancelled()

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
            benchmark_flags=benchmark_flags,
            benchmark_gpu_stats=benchmark_gpu_stats,
            temp_files=temp_files,
            progress_cb=reporter,
            quiet=quiet,
            cancellation_token=cancellation_token,
        )

    # ------------------------------------------------------------------ #
    # Error handling                                                      #
    # ------------------------------------------------------------------ #
    except TranscriptionCancelledError:
        reporter.on_error("Cancelled by user")
        click.secho("\n⚠️ Operation cancelled by user.", fg="yellow", err=True)
        # Cleanup backend resources on cancellation to release GPU memory
        try:
            if hasattr(cli_facade, "backend") and cli_facade.backend is not None:
                cli_facade.backend.close()
                logger.debug("Backend resources released after cancellation")
        except Exception as e:
            logger.warning("Failed to cleanup backend after cancellation: %s", e)
        sys.exit(130)

    except DeviceNotFoundError as exc:
        reporter.on_error(str(exc))
        click.secho(f"\n❌ Device error: {exc}", fg="red", err=True)
        click.secho(
            "💡 Try --device cpu or verify your CUDA/MPS setup", fg="yellow", err=True
        )
        if debug:
            logger.exception("Device error details")
        sys.exit(1)

    except TranscriptionError as exc:
        message = str(exc)
        reporter.on_error(message)
        if task == "transcribe":
            error_label = "Transcription"
        elif task == "translate":
            error_label = "Translation"
        else:
            error_label = task.capitalize()
        click.secho(f"\n❌ {error_label} error: {message}", fg="red", err=True)
        if debug:
            logger.exception("%s error details", task)
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        reporter.on_error(str(exc))
        click.secho(f"\n❌ Unexpected error: {exc}", fg="red", err=True)
        if debug:
            logger.exception("Unexpected error during %s", task)
        else:
            click.secho(
                "💡 Re-run with --debug for more details", fg="yellow", err=True
            )
        sys.exit(1)
    finally:
        for sig, handler in previous_signal_handlers:
            signal.signal(sig, handler)
        if gpu_sampler is not None:
            try:
                gpu_sampler.stop()
            except Exception:  # pragma: no cover - defensive
                pass


# --------------------------------------------------------------------------- #
# Helper: output files + benchmark                                            #
# --------------------------------------------------------------------------- #


def _handle_output_and_benchmarks(
    *,
    task: str,
    audio_file: Path,
    result: dict,
    total_time: float,
    output: Path | None,
    export_format: str,
    export_format_explicit: bool,
    benchmark_enabled: bool,
    benchmark_extra: tuple[str, ...],
    benchmark_flags: dict[str, Any] | None,
    benchmark_gpu_stats: dict[str, Any] | None,
    temp_files: list[Path],
    progress_cb: ProgressCallback | None = None,
    quiet: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> None:
    """Handle file export and benchmark writing.

    Logs debug information about export formats and benchmark collection.

    Args:
        task: Task name ("transcribe" or "translate").
        audio_file: Path to the original audio file.
        result: Result dictionary returned by the backend.
        total_time: Total wall-clock time for the operation in seconds.
        output: Optional explicit output file path when a single format is
            requested by the user.
        export_format: Selected export format ("all", "json", "srt", "txt").
        export_format_explicit: Whether the export format was explicitly chosen
            by the user via CLI flags.
        benchmark_enabled: Whether to collect benchmark metrics.
        benchmark_extra: Additional key=value entries to include in the
            benchmark record.
        benchmark_flags: Mapping of CLI flag names to the values used.
        benchmark_gpu_stats: Aggregated GPU utilisation statistics, if sampled.
        temp_files: List of temporary files to clean up after export.
        progress_cb: Optional progress callback to report export progress.
        quiet: If True, suppress non-essential messages (e.g., benchmark line).
        cancellation_token: Optional cooperative cancellation token.

    Raises:
        TranscriptionCancelledError: If cancellation is requested during export
            or benchmark collection.
    """
    if cancellation_token is not None and cancellation_token.cancelled:
        raise TranscriptionCancelledError("Transcription cancelled by user")

    # Decide which formats to export
    if export_format == "all":
        formats_to_export = ("json", "txt", "srt")
    else:
        formats_to_export = (export_format,)

    logger.debug(
        "_handle_output_and_benchmarks: task=%s, export_format=%s, "
        "formats_to_export=%s, benchmark_enabled=%s",
        task,
        export_format,
        formats_to_export,
        benchmark_enabled,
    )

    # Build a “detailed result” once for all formatters
    detailed_result = {
        task: result["text"],
        "text": result["text"],  # TxtFormatter uses this
        # Prefer the stable-ts key name
        "segments": result.get("segments") or result.get("chunks", []),
        "chunks": result.get("chunks", []),
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
    # Export progress
    try:
        if progress_cb is not None:
            # type: ignore[attr-defined]
            progress_cb.on_export_started(len(formats_to_export))
    except Exception:  # pragma: no cover
        pass

    srt_text_captured: str | None = None
    format_quality_by_format: dict[str, Any] = {}
    for idx, fmt in enumerate(formats_to_export):
        if cancellation_token is not None and cancellation_token.cancelled:
            raise TranscriptionCancelledError("Transcription cancelled by user")
        formatter = FORMATTERS[fmt]
        content = formatter.format(detailed_result)
        ext = formatter.get_file_extension()
        if fmt == "srt":
            srt_text_captured = content
            if benchmark_enabled:
                try:
                    quality_segments = build_quality_segments(detailed_result)
                    logger.debug(
                        "Built %d quality segments for SRT quality scoring",
                        len(quality_segments),
                    )
                    srt_quality = compute_srt_quality(
                        segments=quality_segments,
                        srt_text=srt_text_captured,
                    )
                    logger.debug("SRT quality metrics: %s", srt_quality)
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to compute SRT quality metrics")
                else:
                    format_quality_by_format["srt"] = srt_quality

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
            if cancellation_token is not None and cancellation_token.cancelled:
                raise TranscriptionCancelledError("Transcription cancelled by user")
            output_path.write_text(content, encoding="utf-8")
            content_size = len(content) if isinstance(content, str) else 0
            logger.debug(
                "Saved %s output to: %s (size=%d bytes)",
                fmt.upper(),
                output_path,
                content_size,
            )
            click.secho(f"💾 Saved {fmt.upper()} to: {output_path}", fg="green")
        except OSError as exc:
            click.secho(f"❌ Failed to save {fmt.upper()}: {exc}", fg="red", err=True)
        else:
            try:
                if progress_cb is not None:
                    # type: ignore[attr-defined]
                    # Provide full context to tqdm reporter when single item
                    # using the convention "FMT::/full/path" so it can print
                    # a concise checkmark instead of showing a bar.
                    export_label = f"{fmt.upper()}::{output_path}"
                    progress_cb.on_export_item_done(idx, export_label)
            except Exception:  # pragma: no cover
                pass

    if benchmark_enabled:
        if cancellation_token is not None and cancellation_token.cancelled:
            raise TranscriptionCancelledError("Transcription cancelled by user")
        from insanely_fast_whisper_rocm.benchmarks.collector import BenchmarkCollector

        collector = BenchmarkCollector()
        extra_dict: dict[str, str] | None = None
        if benchmark_extra:
            extra_dict = dict(item.split("=", 1) for item in benchmark_extra)
            logger.debug("Benchmark extra metadata: %s", extra_dict)

        # Merge CLI flags into the config snapshot without overwriting
        # pre-existing config values. This avoids duplicating data across
        # two separate groups ("config" vs "flags").
        merged_config = dict(result.get("config_used") or {})
        if benchmark_flags:
            for k, v in benchmark_flags.items():
                if k not in merged_config:
                    merged_config[k] = v

        # Benchmark details logged by collector
        benchmark_path = collector.collect(
            audio_path=str(audio_file),
            task=task,
            config=merged_config,
            runtime_seconds=result.get("runtime_seconds"),
            total_time=total_time,
            extra=extra_dict,
            gpu_stats=benchmark_gpu_stats,
            format_quality=format_quality_by_format or None,
        )
        # Print benchmark path even when --quiet is set
        click.secho(f"📈 Benchmark saved to: {benchmark_path}", fg="green")

    # ------------------------------------------------------------------ #
    # Cleanup                                                            #
    # ------------------------------------------------------------------ #
    if temp_files:
        cleanup_temp_files(temp_files)
