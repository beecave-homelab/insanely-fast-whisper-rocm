#!/usr/bin/env python
"""User configuration setup script for Insanely Fast Whisper API.

This script copies the .env.example file to the user's configuration
directory to help them get started with custom settings.
It's designed to be run both directly and via PDM.
"""

import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.

    The script is in the 'scripts' directory, one level below project root.
    """
    return Path(__file__).resolve().parent.parent


def get_user_config_dir() -> Path:
    """Return the user configuration directory."""
    return Path.home() / ".config" / "insanely-fast-whisper-rocm"


def get_source_env_file(project_root: Path) -> Path:
    """Return the path to the source .env.example file."""
    return project_root / ".env.example"


def get_user_env_file(user_config_dir: Path) -> Path:
    """Return the path to the user's .env file."""
    return user_config_dir / ".env"


def main() -> None:
    """Manages the creation/update of the user-specific .env file."""
    project_root = get_project_root()
    user_config_dir = get_user_config_dir()
    source_env_file = get_source_env_file(project_root)
    user_env_file = get_user_env_file(user_config_dir)

    print(f"🔧 Attempting to set up user configuration at: {user_env_file}")

    if not source_env_file.exists():
        print(f"❌ ERROR: Source file '{source_env_file}' not found.")
        print("Please ensure '.env.example' exists in the project root.")
        return

    if user_env_file.exists():
        print(f"⚠️ User configuration file already exists at '{user_env_file}'.")
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
        user_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_env_file, user_env_file)
        print(f"\n✅ Successfully copied '{source_env_file}' to '{user_env_file}'.")
        print("\nPlease edit this file to add your specific configurations, such as:")
        print("  - HF_TOKEN (if using gated models like speaker diarization)")
        print("  - Other API keys or custom settings as needed.")
        print(
            "\nROCm / AMD GPUs: if you have an unsupported GPU model "
            "(e.g. RX 6600 / gfx1032),"
        )
        print(
            "you can sometimes patch ROCm compatibility by uncommenting "
            "HSA_OVERRIDE_GFX_VERSION in your .env file."
        )
        print("\nTo find your GPU's GFX target:")
        print("  - rocm_agent_enumerator -name")
        print("  - rocminfo  (look for a GPU agent 'Name: gfxXXXX')")
        print("\nThen choose a supported target from the ROCm compatibility matrix:")
        print(
            "  - https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
        )
        print(
            "\nThe env-var format is MAJOR.MINOR.PATCH derived from the target gfx "
            "string."
        )
        print(
            "For example, forcing gfx1030 corresponds to: "
            "HSA_OVERRIDE_GFX_VERSION=10.3.0"
        )
        print(
            "GPU-to-gfx reference: "
            "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html"
        )
        print(
            "\nNote: The application automatically detects your PyTorch version and "
            "sets the correct allocator configuration (PYTORCH_ALLOC_CONF for >=2.9.0, "
            "PYTORCH_HIP_ALLOC_CONF for <2.9.0). You can customize this in your .env "
            "file if needed."
        )
    except Exception as e:
        print(f"\n❌ An error occurred during setup: {e}")


if __name__ == "__main__":
    main()
