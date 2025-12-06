"""Download a Hugging Face model if it's not already cached or if forced."""

import logging
import sys
from pathlib import Path

import click

# Attempt to import huggingface_hub, provide guidance if not found
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError
except ImportError:
    print(
        "huggingface_hub not found. Please install it: pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

from insanely_fast_whisper_rocm.utils.constants import DEFAULT_MODEL, HF_TOKEN

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger

# Environment variable names
ENV_VAR_MODEL_NAME = "WHISPER_MODEL"
ENV_VAR_HF_TOKEN = "HUGGINGFACE_TOKEN"

# Note: DEFAULT_MODEL and HF_TOKEN are imported from constants.py for
# centralized configuration


def download_model_if_needed(
    model_name: (
        str | None
    ) = None,  # Changed signature to allow None and default to None
    force: bool = False,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    custom_logger: logging.Logger | None = None,
):
    """Ensure a Hugging Face model is present locally, downloading it if necessary.

    If `model_name` is None the module default model is used. Token precedence is
    left to the calling code and the Hugging Face client (an explicit `hf_token`
    argument takes precedence when provided). When `local_files_only` is True the
    function will not attempt network downloads.

    Parameters:
        model_name (str | None): Hugging Face repo ID of the model (e.g. "openai/whisper-large-v3").
                                 If None, the module default model is used.
        force (bool): If True, re-download the model even if it exists in the cache.
        hf_token (str | None): Hugging Face API token to use for authentication; if None,
                               the huggingface_hub library may read environment variables.
        cache_dir (str | Path | None): Directory to use for the HF cache; if None, HF defaults apply.
        local_files_only (bool): If True, restricts resolution to local cache and avoids network access.
        allow_patterns (list[str] | None): If provided, only files matching these patterns will be fetched.
        ignore_patterns (list[str] | None): If provided, files matching these patterns will be excluded.
        custom_logger (logging.Logger | None): Optional logger to use; module logger is used if omitted.

    Returns:
        str: Filesystem path to the downloaded or cached model directory.

    Raises:
        FileNotFoundError: If `local_files_only` is True and the model is not found locally.
        HfHubHTTPError: For HTTP errors returned by the Hugging Face Hub (e.g., 401, 404).
        HFValidationError: If the repository identifier or configuration is invalid.
        OSError: For network or filesystem-related errors during download.
        RuntimeError: For other unexpected runtime errors surfaced during retrieval.
    """
    log = custom_logger if custom_logger else logger

    effective_model_name = model_name

    if not effective_model_name:
        log.info("No model name provided, using centralized default from constants.py")
        effective_model_name = DEFAULT_MODEL

        # Resolve authentication token *lazily* to avoid leaking credentials in
    # contexts (e.g. CI) where the surrounding environment deliberately
    # unsets them.  Only look at environment variables at call-time and never
    # fall back to the module-level HF_TOKEN constant which may have been
    # initialised before the env was sanitized by a test-runner.
    # Respect explicit hf_token argument; otherwise *do not* pull from environment
    # so tests can verify that we don't leak credentials via function parameters.
    if hf_token is not None:
        log.debug("Using Hugging Face token provided as argument.")
    # If hf_token is None we simply pass it through; snapshot_download will
    # internally read relevant environment variables if required.  This prevents
    # the token value from being exposed in call arguments during tests/CI.

    action = "Checking for" if local_files_only else "Downloading"
    log.info("%s model: '%s'", action, effective_model_name)
    if force and not local_files_only:
        log.info("Force re-download enabled.")
    elif force and local_files_only:
        log.warning("Force re-download is ignored when 'local_files_only' is True.")

    try:
        download_path = snapshot_download(
            repo_id=effective_model_name,
            token=hf_token,
            force_download=force and not local_files_only,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            # user_agent can be used for tracking if needed, e.g.:
            # user_agent={"pipeline": "insanely-fast-whisper-rocm/model-downloader"}
        )
        log.info("Model '%s' is available at: %s", effective_model_name, download_path)
        return download_path
    except HFValidationError as e:
        log.error(
            "Invalid model identifier or configuration for '%s': %s",
            effective_model_name,
            e,
        )
        raise
    except HfHubHTTPError as e:
        log.error("HTTP error while accessing model '%s': %s", effective_model_name, e)
        if e.response is not None:
            if e.response.status_code == 401:
                log.error(
                    "Authentication failed. This could be a private/gated model. "
                    "Ensure HUGGINGFACE_TOKEN is set correctly or you have "
                    "access rights."
                )
            elif e.response.status_code == 404:
                log.error(
                    "Model '%s' not found on Hugging Face Hub.", effective_model_name
                )
        raise
    except FileNotFoundError as e:
        log.error("Model '%s' not found in local cache: %s", effective_model_name, e)
        raise
    except (OSError, RuntimeError) as e:
        log.error(
            "An unexpected error occurred with model '%s': %s", effective_model_name, e
        )
        raise


@click.command(
    help="Download or verify a Hugging Face model, typically for Whisper ASR.",
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "-m",
    "--model",
    "model_name_option",
    default=DEFAULT_MODEL,
    show_default=f"centralized default: '{DEFAULT_MODEL}'",
    help="Name of the Hugging Face model (e.g., 'openai/whisper-large-v3').",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-download of the model, ignored if --check_only is used.",
)
@click.option(
    "--hf_token",
    default=HF_TOKEN,
    show_default="centralized configuration",
    help=(
        "Hugging Face API token. If not provided, uses centralized "
        "configuration from constants.py."
    ),
)
@click.option(
    "--cache_dir",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path
    ),
    default=None,
    show_default="Hugging Face default",
    help=(
        "Path to a custom cache directory. If not set, Hugging Face defaults are used."
    ),
)
@click.option(
    "--check_only",
    is_flag=True,
    help=(
        "Only check if the model is cached locally (uses local_files_only), "
        "do not download."
    ),
)
@click.option(
    "--allow_patterns",
    multiple=True,
    help="Patterns for files to include (e.g., '*.bin' '*.json'). Returns a tuple.",
)
@click.option(
    "--ignore_patterns",
    multiple=True,
    help="Patterns for files to exclude (e.g., '*.safetensors'). Returns a tuple.",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose logging (DEBUG level)."
)
def main(
    model_name_option: str,
    force: bool,
    hf_token: str | None,
    cache_dir: Path | None,
    check_only: bool,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
    verbose: bool,
):
    """Main function to handle CLI arguments and trigger model download using Click."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)  # Set root logger level
        logger.setLevel(logging.DEBUG)  # Set our specific logger level
        logger.debug("Verbose logging enabled.")
        # Log actual resolved values for clarity,
        # click handles defaults before this point
        logger.debug("Effective model: %s", model_name_option)
        logger.debug(
            "Force: %s, HF Token: %s, Cache Dir: %s",
            force,
            "Provided" if hf_token else "Not Provided (will check env)",
            cache_dir,
        )
        logger.debug(
            "Check Only: %s, Allow: %s, Ignore: %s",
            check_only,
            allow_patterns,
            ignore_patterns,
        )

    # model_name_option is the parameter from click, assign to model_name
    # for consistency if desired
    model_name = model_name_option

    if not model_name:
        # This case should ideally be caught by click's required=True or a default
        # that doesn't result in None/empty
        # but as a safeguard:
        logger.error(
            "No model specified. Provide one via --model argument or %s "
            "environment variable.",
            ENV_VAR_MODEL_NAME,
        )
        # click.echo(click.get_current_context().get_help(), err=True)
        # Alternative to parser.print_help
        sys.exit(1)

    try:
        logger.info("Processing model: %s", model_name)
        download_path = download_model_if_needed(
            model_name=model_name,
            force=force,
            hf_token=hf_token,
            cache_dir=cache_dir,
            local_files_only=check_only,
            # Convert tuple from click to list if snapshot_download expects list
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            custom_logger=logger,  # Pass the module logger
        )

        if check_only:
            logger.info(
                "Model '%s' successfully checked. Cache location: %s",
                model_name,
                download_path,
            )
        else:
            logger.info(
                "Model '%s' successfully downloaded/verified. Location: %s",
                model_name,
                download_path,
            )
        click.echo(download_path)  # Output path for scripting, use click.echo

    except (HfHubHTTPError, HFValidationError):
        logger.error(
            "Hugging Face Hub error for model '%s'. See logs above.", model_name
        )
        sys.exit(4)
    except (OSError, RuntimeError, ValueError, TypeError) as e:
        logger.critical(
            "Critical error processing model '%s': %s", model_name, e, exc_info=verbose
        )
        sys.exit(5)


if __name__ == "__main__":
    # This allows the script to be run as a module, e.g.:
    # python -m insanely_fast_whisper_rocm.download_hf_model --model openai/whisper-tiny
    main()  # pylint: disable=no-value-for-parameter
