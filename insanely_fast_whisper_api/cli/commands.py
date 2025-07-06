"""CLI commands implementation using Command pattern.

This module contains the implementation of CLI commands that use the facade
to access core ASR functionality, eliminating code duplication.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

from insanely_fast_whisper_api.cli.facade import cli_facade
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.core.formatters import FORMATTERS
from insanely_fast_whisper_api.utils import constants
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
    debug: bool,
    no_timestamps: bool,
    export_format: str,
    export_json: bool,
    export_srt: bool,
    export_txt: bool,
    export_all: bool,
) -> None:
    """
    Transcribe an audio file using Whisper models.

    This command uses the core ASR backend through the CLI facade,
    eliminating code duplication and ensuring consistency.
    """
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
        export_format=export_format,
        export_json=export_json,
        export_srt=export_srt,
        export_txt=export_txt,
        export_all=export_all,
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
    export_format: str,
    export_json: bool,
    export_srt: bool,
    export_txt: bool,
    export_all: bool,
) -> None:
    """
    Translate an audio file to English using Whisper models.

    This command uses the core ASR backend through the CLI facade.
    """
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
        export_format=export_format,
        export_json=export_json,
        export_srt=export_srt,
        export_txt=export_txt,
        export_all=export_all,
    )


def _run_task(**kwargs) -> None:
    """Generic handler for running transcription or translation tasks."""
    debug = kwargs.pop("debug")
    no_timestamps = kwargs.pop("no_timestamps")
    task = kwargs.pop("task")
    audio_file = kwargs.pop("audio_file")
    output = kwargs.pop("output")
    language = kwargs.pop("language")
    export_format = kwargs.pop("export_format", "json")

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
        click.secho(f"📁 Audio file: {audio_file}", fg="blue")

        processed_language = (
            language if language and language.lower() != "none" else None
        )

        click.secho(f"\n⏳ Starting {task_display_name}...", fg="yellow")
        start_time = time.time()

        result = cli_facade.process_audio(
            audio_file_path=audio_file,
            language=processed_language,
            task=task,
            return_timestamps=not no_timestamps,
            **kwargs,
        )

        total_time = time.time() - start_time

        click.secho(f"\n✅ {task_display_name} completed!", fg="green", bold=True)
        click.secho(f"⏱️  Total time: {total_time:.2f}s", fg="yellow")
        click.secho(
            f"🚀 Processing time: {result.get('runtime_seconds', 'N/A')}s", fg="yellow"
        )

        click.secho(f"\n📝 {task_display_name}:", fg="cyan", bold=True)
        click.echo(result["text"])

        # Determine which formats to export
        if export_format == "all":
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
                click.secho(f"\n💾 Results in {fmt.upper()} format saved to: {output_path}", fg="green")
            except (OSError, IOError, UnicodeError) as e:
                click.secho(f"\n❌ Failed to save {fmt.upper()} results: {e}", fg="red", err=True)

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
