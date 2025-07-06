"""Entrypoint for the WebUI.

This file allows the WebUI to be run as a package using the command:
`python -m insanely_fast_whisper_api.webui`
"""

from insanely_fast_whisper_api.webui.app import launch_webui

if __name__ == "__main__":
    launch_webui()  # pylint: disable=no-value-for-parameter
