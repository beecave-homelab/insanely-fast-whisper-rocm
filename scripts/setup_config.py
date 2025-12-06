#!/usr/bin/env python
"""User configuration setup script for Insanely Fast Whisper API.

This script copies the .env.example file to the user's configuration
directory to help them get started with custom settings.
It's designed to be run both directly and via PDM.
"""

import shutil
import sys
from pathlib import Path

# --- Add project root to Python path ---
# This allows the script to import modules from the main application package,
# ensuring it works correctly when run directly or via PDM.
try:
    # Assumes the script is in the 'scripts' directory, one level below project root
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_project_root))

    from insanely_fast_whisper_rocm.utils.constants import (
        PROJECT_ROOT,
        USER_CONFIG_DIR,
        USER_ENV_FILE,
    )
except ImportError as e:
    print(f"❌ Error: Failed to import project constants. {e}")
    print("Please ensure you are running this script from the project root,")
    print("or that the 'insanely_fast_whisper_rocm' package is in your PYTHONPATH.")
    sys.exit(1)


# The source template is always located at the project root.
SOURCE_ENV_FILE = PROJECT_ROOT / ".env.example"


def main():
    """Manages the creation/update of the user-specific .env file."""
    print(f"🔧 Attempting to set up user configuration at: {USER_ENV_FILE}")

    if not SOURCE_ENV_FILE.exists():
        print(f"❌ ERROR: Source file '{SOURCE_ENV_FILE}' not found.")
        print("Please ensure '.env.example' exists in the project root.")
        return

    if USER_ENV_FILE.exists():
        print(f"⚠️ User configuration file already exists at '{USER_ENV_FILE}'.")
        try:
            overwrite_choice = (
                input("Do you want to overwrite it? (yes/no) [no]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            return

        if overwrite_choice not in ["yes", "y"]:
            print("Exiting without overwriting. Your existing configuration is safe.")
            return
        print("Proceeding to overwrite existing configuration...")

    try:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ENV_FILE, USER_ENV_FILE)
        print(f"\n✅ Successfully copied '{SOURCE_ENV_FILE}' to '{USER_ENV_FILE}'.")
        print("\nPlease edit this file to add your specific configurations, such as:")
        print("  - HUGGINGFACE_TOKEN (if using gated models like speaker diarization)")
        print("  - Other API keys or custom settings as needed.")
    except Exception as e:
        print(f"\n❌ An error occurred during setup: {e}")


if __name__ == "__main__":
    main()
