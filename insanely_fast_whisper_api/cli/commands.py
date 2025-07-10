"""CLI commands implementation using Command pattern.

This module contains the implementation of CLI commands that use the facade
to access core ASR functionality, eliminating code duplication.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import click
from click.core import ParameterSource

from insanely_fast_whisper_api.audio.processing import extract_audio_from_video
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


@click.command(
    short_help=(
        "Transcribe audio files [--model, --device, --dtype, --batch-size, "
        "--chunk-length, --language, --output --help]"
    )
)
@click.argument(
    "audio_file", type=click.Path(exists=True, path_type=Path), metavar="AUDIO_FILE"
)
@click.option(
    "--model",
    "-m",
    help="Model name to use for transcription (e.g., distil-whisper/distil-large-v3)",
    show_default=True,
    default=constants.DEFAULT_MODEL,
)
@click.option(
    "--device",
    "-d",
    help="Device for inference (cuda:0, cpu, mps)",
    show_default=True,
    default=constants.DEFAULT_DEVICE,
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default=constants.DEFAULT_DTYPE,
    help="Data type for model inference",
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(constants.MIN_BATCH_SIZE, constants.MAX_BATCH_SIZE),
    default=constants.DEFAULT_BATCH_SIZE,
    help=f"Batch size for processing ({constants.MIN_BATCH_SIZE}-{constants.MAX_BATCH_SIZE})",
    show_default=True,
)
@click.option(
    "--chunk-length",
    "-c",
    default=constants.DEFAULT_CHUNK_LENGTH,
    type=int,
    help="Audio chunk length in seconds",
    show_default=True,
)
@click.option(
    "--language",
    "-l",
    help="Language code (en, fr, de, None=auto)",
    show_default=True,
    default=constants.DEFAULT_LANGUAGE,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Save detailed results to JSON file (default: transcripts/[audio_filename].json)",
)
@click.option(
    "--timestamp-type",
    type=click.Choice(["word", "chunk"]),
    default=constants.DEFAULT_TIMESTAMP_TYPE,
    help="Timestamp granularity when enabled",
    show_default=True,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for troubleshooting",
)
@click.option(
    "--no-timestamps",
    is_flag=True,
    help="Disable timestamp extraction (may fix tensor size errors)",
)
@click.option(
    "--export-format",
    type=click.Choice(["all", "json", "srt", "txt"], case_sensitive=False),
    default="json",
    help="Export format for the transcription.",
    show_default=True,
)
@click.option(
    "--export-json",
    is_flag=True,
    help="[DEPRECATED] Use --export-format json instead.",
    hidden=True,
)
@click.option(
    "--export-srt",
    is_flag=True,
    help="[DEPRECATED] Use --export-format srt instead.",
    hidden=True,
)
@click.option(
    "--export-txt",
    is_flag=True,
    help="[DEPRECATED] Use --export-format txt instead.",
    hidden=True,
)
@click.option(
    "--benchmark-extra",
    multiple=True,
    metavar="KEY=VALUE",
    help="Additional benchmark metrics (can be repeated, KEY=VALUE)",
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Record benchmark metrics (writes JSON to benchmarks/ directory)",
)
@click.option(
    "--export-all",
    is_flag=True,
    help="[DEPRECATED] Use --export-format all instead.",
    hidden=True,
)
def transcribe(
    audio_file: Path,
    model: str,
    device: str,
    dtype: str,
    batch_size: int,
    chunk_length: int,
    language: str,
    output: Optional[Path],
    timestamp_type: str,
    debug: bool,
    no_timestamps: bool,
    export_format: str,
    export_json: bool,
    export_srt: bool,
    export_txt: bool,
    export_all: bool,
    benchmark: bool,
    benchmark_extra: tuple[str, ...],
) -> None:
    """
    Transcribe an audio file using Whisper models.

    This command uses the core ASR backend through the CLI facade,
    eliminating code duplication and ensuring consistency.
    """
    # Determine if --export-format was explicitly provided
    ctx = click.get_current_context()
    export_format_explicit = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(
        task="transcribe",
        audio_file=audio_file,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        chunk_length=chunk_length,
        language=language,
        output=output,
        debug=debug,
        no_timestamps=no_timestamps,
        timestamp_type=timestamp_type,
        export_format=export_format,
        export_json=export_json,
        export_srt=export_srt,
        export_txt=export_txt,
        export_all=export_all,
        benchmark=benchmark,
        benchmark_extra=benchmark_extra,
        export_format_explicit=export_format_explicit,
    )


@click.command(
    short_help=(
        "Translate audio files to English [--model, --device, --dtype, --batch-size, "
        "--chunk-length, --language, --output --help]"
    )
)
@click.argument(
    "audio_file", type=click.Path(exists=True, path_type=Path), metavar="AUDIO_FILE"
)
@click.option(
    "--model",
    "-m",
    help="Model name to use for translation (e.g., distil-whisper/distil-large-v3)",
    show_default=True,
    default=constants.DEFAULT_MODEL,
)
@click.option(
    "--device",
    "-d",
    help="Device for inference (cuda:0, cpu, mps)",
    show_default=True,
    default=constants.DEFAULT_DEVICE,
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default=constants.DEFAULT_DTYPE,
    help="Data type for model inference",
    show_default=True,
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(constants.MIN_BATCH_SIZE, constants.MAX_BATCH_SIZE),
    default=constants.DEFAULT_BATCH_SIZE,
    help=f"Batch size for processing ({constants.MIN_BATCH_SIZE}-{constants.MAX_BATCH_SIZE})",
    show_default=True,
)
@click.option(
    "--chunk-length",
    "-c",
    default=constants.DEFAULT_CHUNK_LENGTH,
    type=int,
    help="Audio chunk length in seconds",
    show_default=True,
)
@click.option(
    "--language",
    "-l",
    help="Language code (en, fr, de, None=auto)",
    show_default=True,
    default=constants.DEFAULT_LANGUAGE,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Save detailed results to JSON file (default: transcripts/[audio_filename].json)",
)
@click.option(
    "--timestamp-type",
    type=click.Choice(["word", "chunk"]),
    default=constants.DEFAULT_TIMESTAMP_TYPE,
    help="Timestamp granularity when enabled",
    show_default=True,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for troubleshooting",
)
@click.option(
    "--no-timestamps",
    is_flag=True,
    help="Disable timestamp extraction (may fix tensor size errors)",
)
@click.option(
    "--export-format",
    type=click.Choice(["all", "json", "srt", "txt"], case_sensitive=False),
    default="json",
    help="Export format for the translation.",
    show_default=True,
)
@click.option(
    "--export-json",
    is_flag=True,
    help="[DEPRECATED] Use --export-format json instead.",
    hidden=True,
)
@click.option(
    "--export-srt",
    is_flag=True,
    help="[DEPRECATED] Use --export-format srt instead.",
    hidden=True,
)
@click.option(
    "--export-txt",
    is_flag=True,
    help="[DEPRECATED] Use --export-format txt instead.",
    hidden=True,
)
@click.option(
    "--benchmark-extra",
    multiple=True,
    metavar="KEY=VALUE",
    help="Additional benchmark metrics (can be repeated, KEY=VALUE)",
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Record benchmark metrics (writes JSON to benchmarks/ directory)",
)
@click.option(
    "--export-all",
    is_flag=True,
    help="[DEPRECATED] Use --export-format all instead.",
    hidden=True,
)
def translate(
    audio_file: Path,
    model: str,
    device: str,
    dtype: str,
    batch_size: int,
    chunk_length: int,
    language: str,
    output: Optional[Path],
    debug: bool,
    no_timestamps: bool,
    timestamp_type: str,
    export_format: str,
    export_json: bool,
    export_srt: bool,
    export_txt: bool,
    export_all: bool,
    benchmark: bool,
    benchmark_extra: tuple[str, ...],
) -> None:
    """
    Translate an audio file to English using Whisper models.

    This command uses the core ASR backend through the CLI facade.
    """
    # Determine if --export-format was explicitly provided
    ctx = click.get_current_context()
    export_format_explicit = (
        ctx.get_parameter_source("export_format") == ParameterSource.COMMANDLINE
    )

    _run_task(
        task="translate",
        audio_file=audio_file,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        chunk_length=chunk_length,
        language=language,
        output=output,
        debug=debug,
        no_timestamps=no_timestamps,
        timestamp_type=timestamp_type,
        export_format=export_format,
        export_json=export_json,
        export_srt=export_srt,
        export_txt=export_txt,
        export_all=export_all,
        benchmark=benchmark,
        benchmark_extra=benchmark_extra,
        export_format_explicit=export_format_explicit,
    )


def _run_task(**kwargs) -> None:
    """Generic handler for running transcription or translation tasks."""
    debug = kwargs.pop("debug")
    no_timestamps = kwargs.pop("no_timestamps")
    timestamp_type = kwargs.pop("timestamp_type")
    # Determine if --no-timestamps was explicitly provided
    ctx_internal = click.get_current_context(silent=True)
    no_ts_explicit = False
    if ctx_internal is not None:
        from click.core import ParameterSource as _PS

        no_ts_explicit = (
            ctx_internal.get_parameter_source("no_timestamps") == _PS.COMMANDLINE
        )
    task = kwargs.pop("task")
    audio_file = kwargs.pop("audio_file")
    output = kwargs.pop("output")
    language = kwargs.pop("language")
    export_format = kwargs.pop("export_format", "json")
    benchmark_enabled = kwargs.pop("benchmark", False)
    benchmark_extra = kwargs.pop("benchmark_extra", ())

    # Auto-enable no_timestamps during benchmarking unless explicitly provided
    if benchmark_enabled and not no_ts_explicit:
        no_timestamps = True
    export_format_explicit = kwargs.pop("export_format_explicit", False)

    # Handle deprecated flags
    deprecated_flags = {
        "export_json": "json",
        "export_srt": "srt",
        "export_txt": "txt",
        "export_all": "all",
    }
    for flag, format_val in deprecated_flags.items():
        if kwargs.pop(flag, False):
            export_format = format_val
            click.secho(
                f"Warning: --{flag.replace('_', '-')} is deprecated. Use --export-format {format_val} instead.",
                fg="yellow",
            )
            break

    task_display_name = task.capitalize()

    try:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("insanely_fast_whisper_api").setLevel(logging.DEBUG)
            click.secho("🐛 Debug mode enabled", fg="yellow")

        click.secho(
            f"\n🎵 {constants.API_TITLE} v{constants.API_VERSION}", fg="cyan", bold=True
        )
        temp_files: list[str] = []  # Track temporary files we create
        # Detect if input is a video file (simple extension check)
        video_exts = {".mp4", ".mkv", ".webm", ".mov"}
        if audio_file.suffix.lower() in video_exts:
            click.secho("🎞️ Detected video input – extracting audio...", fg="yellow")
            try:
                audio_extracted_path = Path(extract_audio_from_video(str(audio_file)))
                audio_file = audio_extracted_path
                temp_files.append(str(audio_extracted_path))
                click.secho(
                    f"✅ Audio extracted to temporary file: {audio_file}", fg="green"
                )
            except RuntimeError as conv_err:
                click.secho(
                    f"❌ Video conversion failed: {conv_err}", fg="red", err=True
                )
                sys.exit(1)

        click.secho(f"📁 Audio file: {audio_file}", fg="blue")

        processed_language = (
            language if language and language.lower() != "none" else None
        )

        click.secho(f"\n⏳ Starting {task_display_name}...", fg="yellow")
        start_time = time.time()

        # Benchmark support
        collector = None
        if benchmark_enabled:
            try:
                from insanely_fast_whisper_api.utils.benchmark import (
                    BenchmarkCollector,
                )

                collector = BenchmarkCollector()
            except Exception:  # pragma: no cover
                collector = None
        if collector:
            collector.start()

        # Determine return_timestamps_value based on flags
        if no_timestamps:
            return_timestamps_value = False
        else:
            return_timestamps_value = "word" if timestamp_type == "word" else True

        result = cli_facade.process_audio(
            audio_file_path=audio_file,
            language=processed_language,
            task=task,
            return_timestamps_value=return_timestamps_value,
            **kwargs,
        )

        total_time = time.time() - start_time

        # Save benchmark file if requested
        # parse extra into dict
        extra_dict: Dict[str, str] | None = None
        if benchmark_extra:
            extra_dict = {}
            for item in benchmark_extra:
                if "=" in item:
                    key, val = item.split("=", 1)
                    extra_dict[key.strip()] = val.strip()
        benchmark_path: Path | None = None
        if collector:
            try:
                benchmark_path = collector.collect(
                    audio_path=str(audio_file),
                    task=task,
                    config=result.get("config_used"),
                    runtime_seconds=result.get("runtime_seconds"),
                    total_time=total_time,
                    extra=extra_dict,
                )
            except Exception as bench_err:  # pragma: no cover
                click.secho(f"⚠️  Failed to save benchmark: {bench_err}", fg="yellow")

        click.secho(f"\n📝 {task_display_name}:", fg="cyan", bold=True)
        click.echo(result["text"])

        # Determine which formats to export
        # Decide whether to skip exporting transcripts when benchmarking
        # Auto-enable no_timestamps during benchmarking, unless user explicitly set it
        if benchmark_enabled and not no_ts_explicit:
            no_timestamps = True

        if benchmark_enabled and not export_format_explicit:
            formats_to_export = []
        elif export_format == "all":
            formats_to_export = ["json", "srt", "txt"]
        else:
            formats_to_export = [export_format]

        # Prepare detailed result structure once
        detailed_result = {
            task: result["text"],
            "text": result["text"],  # For TxtFormatter
            "chunks": result.get("chunks", []),
            "metadata": {
                "audio_file": str(audio_file.resolve()),
                "total_time_seconds": round(total_time, 2),
                "processing_time_seconds": result.get("runtime_seconds"),
                "config_used": result.get("config_used", {}),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # Export each requested format
        for fmt in formats_to_export:
            formatter = FORMATTERS.get(fmt)
            if not formatter:
                click.secho(f"\n❌ Unknown format: {fmt}", fg="red", err=True)
                continue

            content = formatter.format(detailed_result)
            file_extension = formatter.get_file_extension()

            if output:
                # If an explicit output file is given, use it only for the primary format
                if fmt == export_format and export_format != "all":
                    output_path = Path(output)
                else:
                    # For 'all' or secondary formats, create a sibling file
                    output_path = Path(output).with_suffix(f".{file_extension}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Default directory structure
                if fmt == "json":
                    output_dir = Path(constants.DEFAULT_TRANSCRIPTS_DIR)
                else:
                    output_dir = Path(f"transcripts-{fmt}")

                output_dir.mkdir(exist_ok=True)
                strategy = StandardFilenameStrategy()
                filename_gen = FilenameGenerator(strategy=strategy)
                output_filename_str = filename_gen.create_filename(
                    audio_path=str(audio_file.absolute()),
                    task=TaskType(task),
                    extension=file_extension,
                )
                output_path = output_dir / output_filename_str

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                click.secho(
                    f"\n💾 Results in {fmt.upper()} format saved to: {output_path}",
                    fg="green",
                )
            except (OSError, IOError, UnicodeError) as e:
                click.secho(
                    f"\n❌ Failed to save {fmt.upper()} results: {e}",
                    fg="red",
                    err=True,
                )

        # Cleanup any temp files we generated
        if temp_files:
            cleanup_temp_files(temp_files)

        # Print benchmark path at bottom for visibility
        if benchmark_path:
            click.secho(f"\n📈 Benchmark saved to: {benchmark_path}", fg="green")

        chunks = result.get("chunks")
        if chunks:
            click.secho(f"\n📊 Generated {len(chunks)} chunks", fg="blue")

    except DeviceNotFoundError as e:
        click.secho(f"\n❌ Device Error: {e}", fg="red", err=True)
        click.secho(
            "💡 Try using --device cpu or check your CUDA/MPS installation",
            fg="yellow",
            err=True,
        )
        if debug:
            logger.exception("Device error details")
        sys.exit(1)

    except TranscriptionError as e:
        click.secho(f"\n❌ {task_display_name} Error: {e}", fg="red", err=True)
        if debug:
            logger.exception("%s error details", task_display_name)
        sys.exit(1)

    except (OSError, IOError, ValueError, TypeError, RuntimeError) as e:
        click.secho(f"\n❌ Unexpected error: {e}", fg="red", err=True)
        if debug:
            logger.exception("Unexpected error during %s", task)
        else:
            click.secho(
                "💡 Use --debug flag for more detailed error information",
                fg="yellow",
                err=True,
            )
        sys.exit(1)
