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
    )


def _run_task(**kwargs):
    """Generic handler for running transcription or translation tasks."""
    task = kwargs.pop("task")
    audio_file = kwargs.pop("audio_file")
    debug = kwargs.pop("debug")
    no_timestamps = kwargs.pop("no_timestamps")
    output = kwargs.pop("output")
    language = kwargs.pop("language")

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

        if output is None:
            transcripts_dir = Path(constants.DEFAULT_TRANSCRIPTS_DIR)
            transcripts_dir.mkdir(exist_ok=True)
            strategy = StandardFilenameStrategy()
            filename_gen = FilenameGenerator(strategy=strategy)
            output_filename_str = filename_gen.create_filename(
                audio_path=str(audio_file.absolute()),
                task=TaskType(task),
                extension="json",
            )
            output = transcripts_dir / output_filename_str
        else:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)

        try:
            detailed_result = {
                task: result["text"],
                "chunks": result.get("chunks", []),
                "metadata": {
                    "audio_file": str(audio_file),
                    "total_time_seconds": round(total_time, 2),
                    "processing_time_seconds": result.get("runtime_seconds"),
                    "config_used": result.get("config_used", {}),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
            with open(output, "w", encoding="utf-8") as f:
                json.dump(detailed_result, f, indent=2, ensure_ascii=False)
            click.secho(f"\n💾 Results saved to: {output}", fg="green")

        except (OSError, IOError, UnicodeError) as e:
            click.secho(f"\n❌ Failed to save results: {e}", fg="red", err=True)
            sys.exit(1)

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
