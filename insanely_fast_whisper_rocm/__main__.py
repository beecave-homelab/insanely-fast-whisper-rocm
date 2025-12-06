"""Entry point for running the package as a module.

This module allows running the FastAPI application using:
python -m insanely_fast_whisper_rocm
"""

import logging
import logging.config
import os
import time
from pathlib import Path

import click
import yaml

from insanely_fast_whisper_rocm.utils import constants

try:
    import uvicorn
except ImportError:
    # uvicorn is only needed when running the server, not for all imports
    # This allows other parts of the package to import __main__ if necessary
    # without uvicorn being a hard dependency for all CLI commands.
    uvicorn = None


def setup_timezone():
    """Set the timezone for the application based on constants.APP_TIMEZONE."""
    try:
        os.environ["TZ"] = constants.APP_TIMEZONE
        time.tzset()
        logging.info(
            "Timezone set to: %s (%s) using APP_TIMEZONE='%s'",
            time.tzname[0],
            time.tzname[1],
            constants.APP_TIMEZONE,
        )
    except (TypeError, OSError, IndexError) as e:
        logging.warning(
            "Could not set timezone using APP_TIMEZONE='%s': %s. Using system default.",
            constants.APP_TIMEZONE,
            e,
        )


def load_logging_config(debug: bool = False) -> dict:
    """Load and configure logging from YAML file.

    Args:
        debug: Whether to enable debug logging levels in the YAML config.

    Returns:
        dict: The configured logging dictionary.
    """
    # Ensure timezone is set before loading config
    setup_timezone()

    # Load the YAML config
    config_path = Path(__file__).parent / "logging_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Adjust log levels based on debug flag
    if debug:
        config["root"]["level"] = "DEBUG"
        for logger_config in config.get("loggers", {}).values():
            logger_config["level"] = "DEBUG"

    return config


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    short_help="Starts the Insanely Fast Whisper API server.",
)
@click.option(
    "--host",
    default=constants.API_HOST,
    show_default=True,
    help="The host to bind the server to.",
)
@click.option(
    "--port",
    type=int,
    default=constants.API_PORT,
    show_default=True,
    help="The port to bind the server to.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of worker processes for Uvicorn.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"], case_sensitive=False
    ),
    default=constants.LOG_LEVEL.lower(),
    show_default=True,
    help="Set the Uvicorn log level.",
)
@click.option(
    "--reload/--no-reload",
    default=False,
    show_default=True,
    help="Enable/disable auto-reloading on code changes.",
)
@click.option(
    "--ssl-keyfile",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to SSL key file.",
)
@click.option(
    "--ssl-certfile",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to SSL certificate file.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Enable debug mode (sets YAML log config to DEBUG, implies --log-level "
        "debug if not set)."
    ),
)
@click.version_option(version=constants.API_VERSION, prog_name=constants.API_TITLE)
def main(
    host: str,
    port: int,
    workers: int,
    log_level: str,
    reload: bool,
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    debug: bool,
):
    """
    Start the application HTTP server with Uvicorn using the provided runtime options.
    
    Parameters:
        host (str): Hostname or IP address to bind the server to.
        port (int): TCP port to listen on.
        workers (int): Number of worker processes to run.
        log_level (str): Base logging level; may be overridden to "debug" when `debug` is True and matches the default level.
        reload (bool): Enable Uvicorn auto-reload for development.
        ssl_keyfile (str | None): Path to the SSL key file to enable HTTPS, or None to disable SSL.
        ssl_certfile (str | None): Path to the SSL certificate file to enable HTTPS, or None to disable SSL.
        debug (bool): When True, increase logging verbosity in the YAML configuration and may set Uvicorn's level to "debug".
    
    Raises:
        click.exceptions.Exit: If Uvicorn is not installed (exits with code 1).
    """
    if uvicorn is None:
        click.secho(
            (
                "Uvicorn is not installed. Please install it to run the server: "
                "pip install uvicorn"
            ),
            fg="red",
            err=True,
        )
        raise click.exceptions.Exit(1)

    # If debug flag is set, and log_level is default 'info', upgrade
    # log_level to 'debug'
    effective_log_level = log_level
    if debug and log_level == constants.LOG_LEVEL.lower():
        effective_log_level = "debug"

    # Load and apply logging configuration from YAML
    # The `debug` flag here controls the verbosity within the YAML structure itself.
    log_config_dict = load_logging_config(debug=debug)
    logging.config.dictConfig(log_config_dict)

    # Get logger after configuration
    # This logger is for the application logs once Uvicorn hands over control or for
    # Uvicorn's own logs if configured via log_config_dict
    logger = logging.getLogger("insanely_fast_whisper_rocm")

    # Prettify startup messages
    click.echo()
    click.secho(
        f"🚀 Starting {constants.API_TITLE} v{constants.API_VERSION}",
        fg="cyan",
        bold=True,
    )
    protocol = "https" if ssl_keyfile and ssl_certfile else "http"
    click.secho(
        f"🔗 Listening on: {protocol}://{host}:{port}",
        fg="blue",
    )

    if workers > 1:
        click.secho(f"⚙️  Running with {workers} worker processes.", fg="magenta")

    if reload:
        click.secho("🔄 Auto-reload is enabled (for development).", fg="yellow")

    if ssl_keyfile and ssl_certfile:
        click.secho("🔒 SSL is enabled.", fg="green")
        click.secho(f"   Keyfile: {ssl_keyfile}", fg="green")
        click.secho(f"   Certfile: {ssl_certfile}", fg="green")

    if debug:
        click.secho(
            f"🐛 Debug mode is ON (YAML log levels set to DEBUG, "
            f"Uvicorn log level: {effective_log_level}).",
            fg="bright_red",
        )
    else:
        click.secho(f"🪵  Uvicorn log level: {effective_log_level}", fg="green")

    click.echo("\n✨ Uvicorn is now starting up... (Press CTRL+C to quit)")
    # The logger.info below is less critical now as click.secho provides more visible
    # startup info. However, it can be useful if logs are being piped to a file where
    # click's color codes might not render.
    logger.info(
        "Uvicorn server configured to run on %s:%s with log_level='%s'. Reload: %s",
        host,
        port,
        effective_log_level,
        reload,
    )

    uvicorn.run(
        "insanely_fast_whisper_rocm.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=effective_log_level,  # This controls Uvicorn's direct logging
        reload=reload,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_config=log_config_dict,  # Pass the loaded YAML config to Uvicorn
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter