"""WebUI application logic for Insanely Fast Whisper API.

This module contains the core function for launching the Gradio WebUI.
"""

import logging
import sys

import click

from insanely_fast_whisper_api.utils.constants import (
    DEFAULT_DEMUCS,
    DEFAULT_MODEL,
    DEFAULT_STABILIZE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
)
from insanely_fast_whisper_api.utils.download_hf_model import download_model_if_needed
from insanely_fast_whisper_api.webui.ui import create_ui_components

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_api.webui.app")


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to. Default is 0.0.0.0 (all interfaces).",
)
@click.option(
    "--port",
    default=7860,
    type=int,
    help="Port to bind the server to. Default is 7860.",
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public URL. This is useful for running on Colab or sharing with others.",
)
@click.option(
    "--model",
    help="Whisper model to use. If not specified, uses the WHISPER_MODEL env var or default.",
)
@click.option(
    "--vad-threshold",
    default=DEFAULT_VAD_THRESHOLD,
    type=float,
    help="Voice Activity Detection threshold (0-1) used when --vad is enabled.",
)
@click.option(
    "--vad/--no-vad",
    default=DEFAULT_VAD,
    help="Enable voice activity detection when stabilizing timestamps.",
)
@click.option(
    "--demucs/--no-demucs",
    default=DEFAULT_DEMUCS,
    help="Use Demucs for noise reduction when stabilizing timestamps.",
)
@click.option(
    "--stabilize/--no-stabilize",
    default=DEFAULT_STABILIZE,
    help="Enable word-level timestamp stabilization.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging.",
)
def launch_webui(
    host: str,
    port: int,
    share: bool,
    model: str,
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    debug: bool,
):
    """Launch the Insanely Fast Whisper WebUI."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    logger.info("Starting Insanely Fast Whisper WebUI...")

    # Ensure model is downloaded/verified before UI setup
    logger.info("Attempting to download/verify Whisper model for WebUI...")
    download_model_if_needed(model_name=model, custom_logger=logger)
    logger.info("Model download/verification process for WebUI complete.")

    # Determine which model value should be shown in the UI
    ui_default_model = model if model else DEFAULT_MODEL

    # Create the interface with CLI-provided defaults so checkboxes reflect flags
    iface = create_ui_components(
        default_model=ui_default_model,
        default_stabilize=stabilize,
        default_demucs=demucs,
        default_vad=vad,
        default_vad_threshold=vad_threshold,
    )

    # Launch the interface
    logger.info("Launching WebUI on %s:%s", host, port)
    iface.launch(
        server_name=host,
        server_port=port,
        share=share,
    )
