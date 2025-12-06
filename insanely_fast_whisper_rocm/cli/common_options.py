"""Shared Click option decorator for audio processing commands.

This module defines a single decorator, ``audio_options``, that injects the
full set of Click arguments/options required by both the *transcribe* and
*translate* commands.  Keeping them in one place ensures consistency and
eliminates several hundred lines of duplicate option declarations in
``commands.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click

from insanely_fast_whisper_rocm.utils import constants

# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def audio_options(func: Callable[..., None]) -> Callable[..., None]:  # noqa: D401
    """Attach common audio-processing CLI options to ``func``.

    The decorator adds the same set of arguments/options that were previously
<<<<<<< HEAD:insanely_fast_whisper_api/cli/common_options.py
    duplicated for both *transcribe* and *translate* commands.  It returns the
    wrapped function so it can be stacked beneath ``@click.command``.
=======
    duplicated for both ``transcribe`` and ``translate`` commands.

    Returns:
        Callable[..., None]: The wrapped function, so the decorator can be
        stacked beneath ``@click.command``.
>>>>>>> dev:insanely_fast_whisper_rocm/cli/common_options.py
    """
    options: list[Callable[[Callable[..., None]], Callable[..., None]]] = [
        # Positional argument
        click.argument(
            "audio_file",
            type=click.Path(exists=True, path_type=Path),
            metavar="AUDIO_FILE",
        ),
        # Model / device / dtype / batching
        click.option(
            "--model",
            "-m",
            help="Model name to use (e.g., distil-whisper/distil-large-v3)",
            show_default=True,
            default=constants.DEFAULT_MODEL,
        ),
        click.option(
            "--device",
            "-d",
            help="Device for inference (cuda:0, cpu, mps)",
            show_default=True,
            default=constants.DEFAULT_DEVICE,
        ),
        click.option(
            "--dtype",
            type=click.Choice(["float16", "float32"]),
            default=constants.DEFAULT_DTYPE,
            help="Data type for model inference",
            show_default=True,
        ),
        click.option(
            "--batch-size",
            "-b",
            type=click.IntRange(constants.MIN_BATCH_SIZE, constants.MAX_BATCH_SIZE),
            default=constants.DEFAULT_BATCH_SIZE,
            help=(
                "Batch size for processing "
                f"({constants.MIN_BATCH_SIZE}-{constants.MAX_BATCH_SIZE})"
            ),
<<<<<<< HEAD:insanely_fast_whisper_api/cli/common_options.py
=======
            show_default=True,
        ),
        click.option(
            "--progress-group-size",
            type=click.IntRange(1, 256),
            default=constants.DEFAULT_PROGRESS_GROUP_SIZE,
            help=(
                "Chunks per progress update (higher = fewer UI updates). "
                "Independent of model batch size."
            ),
>>>>>>> dev:insanely_fast_whisper_rocm/cli/common_options.py
            show_default=True,
        ),
        click.option(
            "--chunk-length",
            "-c",
            default=constants.DEFAULT_CHUNK_LENGTH,
            type=int,
            help="Audio chunk length in seconds",
            show_default=True,
        ),
        click.option(
            "--language",
            "-l",
            help="Language code (en, fr, de, None=auto)",
            show_default=True,
            default=constants.DEFAULT_LANGUAGE,
        ),
        click.option(
            "--output",
            "-o",
            type=click.Path(path_type=Path, dir_okay=False),
            help=(
                "Save detailed results to JSON file (default: "
                "transcripts/[audio_filename].json)"
            ),
        ),
        click.option(
            "--timestamp-type",
            type=click.Choice(["word", "chunk"]),
            default=constants.DEFAULT_TIMESTAMP_TYPE,
            help="Timestamp granularity when enabled",
            show_default=True,
        ),
        # Stable-ts related options
        click.option(
            "--stabilize/--no-stabilize",
            default=constants.DEFAULT_STABILIZE,
            help="Post-process results with stable-ts for improved timestamps",
            show_default=True,
        ),
        click.option(
            "--demucs/--no-demucs",
            default=constants.DEFAULT_DEMUCS,
            help="Enable Demucs-based noise reduction in stable-ts",
            show_default=True,
        ),
        click.option(
            "--vad/--no-vad",
            default=constants.DEFAULT_VAD,
            help="Enable Voice Activity Detection (VAD) in stable-ts",
            show_default=True,
        ),
        click.option(
            "--vad-threshold",
            type=click.FloatRange(0.0, 1.0),
            default=constants.DEFAULT_VAD_THRESHOLD,
            help="VAD probability threshold used when --vad is enabled",
            show_default=True,
        ),
        click.option(
            "--debug",
            is_flag=True,
            help="Enable debug logging for troubleshooting",
        ),
        click.option(
            "--progress/--no-progress",
            default=True,
            help=(
                "Show Rich progress UI (auto-disabled on non-TTY). "
                "Use --no-progress to disable."
            ),
            show_default=True,
        ),
        click.option(
            "--no-timestamps",
            is_flag=True,
            help="Disable timestamp extraction (may fix tensor size errors)",
        ),
        click.option(
            "--export-format",
            type=click.Choice(["all", "json", "srt", "txt"], case_sensitive=False),
            default="json",
            help="Export format for the output.",
            show_default=True,
        ),
        # Deprecated flags (keep for backward-compatibility but hidden from --help)
        click.option(
            "--export-json",
            is_flag=True,
            help="[DEPRECATED] Use --export-format json instead.",
            hidden=True,
        ),
        click.option(
            "--export-srt",
            is_flag=True,
            help="[DEPRECATED] Use --export-format srt instead.",
            hidden=True,
        ),
        click.option(
            "--export-txt",
            is_flag=True,
            help="[DEPRECATED] Use --export-format txt instead.",
            hidden=True,
        ),
        click.option(
            "--benchmark-extra",
            multiple=True,
            metavar="KEY=VALUE",
            help="Additional benchmark metrics (can be repeated, KEY=VALUE)",
        ),
        click.option(
            "--benchmark",
            is_flag=True,
            help="Record benchmark metrics (writes JSON to benchmarks/ directory)",
        ),
        click.option(
            "--export-all",
            is_flag=True,
            help="[DEPRECATED] Use --export-format all instead.",
            hidden=True,
        ),
        click.option(
            "--quiet",
            is_flag=True,
            help=(
                "Minimize console output: show only the progress bar and the final "
                "saved output path(s)."
            ),
        ),
    ]

    # Apply in reverse order so the options appear in the original sequence.
    for opt in reversed(options):
        func = opt(func)
    return func
