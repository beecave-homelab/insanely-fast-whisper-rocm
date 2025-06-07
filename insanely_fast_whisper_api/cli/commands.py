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

from insanely_fast_whisper_api.utils import constants
from insanely_fast_whisper_api.core.errors import (
    DeviceNotFoundError,
    TranscriptionError,
)
from insanely_fast_whisper_api.cli.facade import cli_facade
from insanely_fast_whisper_api.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)


@click.command(
    short_help=(
        "Transcribe audio files [--model, --device, --dtype, --batch-size, "
        "--better-transformer, --chunk-length, --language, --output --help]"
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
    "--better-transformer/--no-better-transformer",
    default=constants.DEFAULT_BETTER_TRANSFORMER,
    help="Use BetterTransformer for faster inference",
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
    better_transformer: bool,
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
    try:
        # Set up debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("insanely_fast_whisper_api").setLevel(logging.DEBUG)
            click.secho("🐛 Debug mode enabled", fg="yellow")

        # Display header
        click.secho(
            f"\n🎵 {constants.API_TITLE} v{constants.API_VERSION}", fg="cyan", bold=True
        )
        click.secho(f"📁 Audio file: {audio_file}", fg="blue")

        # Debug: Show file details
        if debug:
            click.secho(
                f"🔍 Audio file path (absolute): {audio_file.absolute()}", fg="yellow"
            )
            click.secho(f"🔍 Audio file exists: {audio_file.exists()}", fg="yellow")
            click.secho(
                f"🔍 Audio file size: {audio_file.stat().st_size if audio_file.exists() else 'N/A'} bytes",
                fg="yellow",
            )
            click.secho(f"🔍 Working directory: {Path.cwd()}", fg="yellow")

        # Handle language parameter
        processed_language = None if language.lower() == "none" else language

        # Debug: Show configuration
        if debug:
            click.secho("\n🔍 Configuration:", fg="yellow")
            click.secho(f"  Model: {model}", fg="yellow")
            click.secho(f"  Device: {device}", fg="yellow")
            click.secho(f"  Dtype: {dtype}", fg="yellow")
            click.secho(f"  Batch size: {batch_size}", fg="yellow")
            click.secho(f"  Better transformer: {better_transformer}", fg="yellow")
            click.secho(f"  Chunk length: {chunk_length}", fg="yellow")
            click.secho(f"  Language: {processed_language}", fg="yellow")
            click.secho(f"  Timestamps: {not no_timestamps}", fg="yellow")

        # Start timing
        start_time = time.time()

        # Use the facade to perform transcription
        result = cli_facade.transcribe_audio(
            audio_file_path=audio_file,
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            better_transformer=better_transformer,
            chunk_length=chunk_length,
            language=processed_language,
            task="transcribe",
            return_timestamps=not no_timestamps,
        )

        # Calculate total time
        total_time = time.time() - start_time

        # Display results
        click.secho("\n✅ Transcription completed!", fg="green", bold=True)
        click.secho(f"⏱️  Total time: {total_time:.2f}s", fg="yellow")
        click.secho(
            f"🚀 Processing time: {result.get('runtime_seconds', 'N/A')}s", fg="yellow"
        )

        # Display transcription text
        click.secho("\n📝 Transcription:", fg="cyan", bold=True)
        click.echo(result["text"])

        # Prepare output file path
        if output is None:
            # Create default output path in transcripts directory
            transcripts_dir = Path(constants.DEFAULT_TRANSCRIPTS_DIR)
            transcripts_dir.mkdir(exist_ok=True)

            # Instantiate filename generator
            strategy = StandardFilenameStrategy()
            filename_gen = FilenameGenerator(strategy=strategy)

            # Generate filename using the new utility
            output_filename_str = filename_gen.create_filename(
                audio_path=str(audio_file.absolute()),
                task=TaskType.TRANSCRIBE,
                extension="json",
            )
            output = transcripts_dir / output_filename_str
        else:
            # If output path is provided, ensure parent directory exists
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        try:
            # Prepare detailed output
            detailed_result = {
                "transcription": result["text"],
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

        # Display chunks summary if available
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
        click.secho(f"\n❌ Transcription Error: {e}", fg="red", err=True)
        if debug:
            logger.exception("Transcription error details")
        sys.exit(1)

    except (OSError, IOError, ValueError, TypeError, RuntimeError) as e:
        click.secho(f"\n❌ Unexpected error: {e}", fg="red", err=True)
        if debug:
            logger.exception("Unexpected error during transcription")
        else:
            click.secho(
                "💡 Use --debug flag for more detailed error information",
                fg="yellow",
                err=True,
            )
        sys.exit(1)
