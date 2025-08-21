#!/usr/bin/env python3
"""CLI tool to convert a Whisper transcription JSON file to TXT and/or SRT formats.

Usage:
  python scripts/convert_json_to_txt_srt.py --input input.json --output-dir out/ --format txt --format srt [--debug]

Requires: insanely_fast_whisper_api in PYTHONPATH or installed.
"""

import json
import logging
import os
import sys

import click

# Import formatters from core
try:
    from insanely_fast_whisper_api.core.formatters import SrtFormatter, TxtFormatter
except ImportError:
    print(
        "Error: Could not import formatters. Make sure insanely_fast_whisper_api is installed or PYTHONPATH is set."
    )
    sys.exit(1)

FORMATTER_MAP = {
    "txt": TxtFormatter,
    "srt": SrtFormatter,
}


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input JSON file.",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to write output files (default: same as input).",
)
@click.option(
    "--format",
    "-f",
    "formats",
    multiple=True,
    type=click.Choice(["txt", "srt"]),
    required=True,
    help="Output format(s): txt, srt. Can be specified multiple times.",
)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cli(input_path, output_dir, formats, debug):
    """Convert a Whisper JSON result to TXT and/or SRT."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("convert_json_to_txt_srt")

    logger.debug(f"Input file: {input_path}")
    logger.debug(f"Output dir: {output_dir}")
    logger.debug(f"Formats: {formats}")

    # Load JSON
    try:
        with open(input_path, encoding="utf-8") as f:
            result = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    for fmt in formats:
        formatter = FORMATTER_MAP[fmt]
        try:
            output_str = formatter.format(result)
        except Exception as e:
            logger.error(f"Failed to convert to {fmt}: {e}")
            continue
        out_path = os.path.join(output_dir, f"{basename}.{fmt}")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output_str)
            logger.info(f"Wrote {fmt.upper()} to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write {fmt} file: {e}")


if __name__ == "__main__":
    cli()
