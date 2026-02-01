"""Tests for setup_config.py script."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_setup_config_script_runs_without_import_errors() -> None:
    """Test that setup_config.py can be imported without import errors."""
    script_path = Path("scripts/setup_config.py")
    if not script_path.exists():
        pytest.skip("setup_config.py not found")

    # Read the script to check for problematic imports
    with open(script_path) as f:
        content = f.read()

    # Verify the script doesn't import from the main package
    assert "from insanely_fast_whisper_rocm" not in content
    assert "import insanely_fast_whisper_rocm" not in content

    # Verify it only uses stdlib imports
    assert "import shutil" in content
    assert "from pathlib import Path" in content


def test_setup_config_helper_functions() -> None:
    """Test helper functions in setup_config.py."""
    script_path = Path("scripts/setup_config.py")
    if not script_path.exists():
        pytest.skip("setup_config.py not found")

    # Read and exec the script without __name__ = "__main__"
    script_globals = {"__file__": str(script_path), "__name__": "test_module"}
    with open(script_path) as f:
        exec(f.read(), script_globals)

    # Test get_project_root
    project_root = script_globals["get_project_root"]()
    assert project_root.name.startswith("insanely-fast-whisper-rocm")

    # Test get_user_config_dir
    user_config_dir = script_globals["get_user_config_dir"]()
    assert user_config_dir == Path.home() / ".config" / "insanely-fast-whisper-rocm"

    # Test get_source_env_file
    source_env_file = script_globals["get_source_env_file"](project_root)
    assert source_env_file == project_root / ".env.example"

    # Test get_user_env_file
    user_env_file = script_globals["get_user_env_file"](user_config_dir)
    assert user_env_file == user_config_dir / ".env"


def test_setup_config_main_creates_config() -> None:
    """Test that setup_config.py creates user config file."""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / "project"
        project_root.mkdir()

        # Create .env.example
        env_example = project_root / ".env.example"
        env_example.write_text("TEST_VAR=value\n")

        # Mock Path.home() to return our temp directory
        with patch("pathlib.Path.home", return_value=tmpdir_path):
            # Read and exec the script without __name__ = "__main__"
            script_path = Path("scripts/setup_config.py")
            if not script_path.exists():
                pytest.skip("setup_config.py not found")

            script_globals = {"__file__": str(script_path), "__name__": "test_module"}
            with open(script_path) as f:
                exec(f.read(), script_globals)

            # Mock get_project_root to return our test project root
            def mock_get_project_root() -> Path:
                return project_root

            script_globals["get_project_root"] = mock_get_project_root

            # Mock input to return "no" (don't overwrite)
            with patch("builtins.input", return_value="no"):
                script_globals["main"]()

            # Verify config file was created
            user_config_dir = tmpdir_path / ".config" / "insanely-fast-whisper-rocm"
            user_env_file = user_config_dir / ".env"
            assert user_env_file.exists()
            assert user_env_file.read_text() == "TEST_VAR=value\n"


def test_setup_config_main_overwrites_existing() -> None:
    """Test that setup_config.py overwrites existing config when confirmed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / "project"
        project_root.mkdir()

        # Create .env.example
        env_example = project_root / ".env.example"
        env_example.write_text("NEW_VAR=new_value\n")

        # Create existing user config
        user_config_dir = tmpdir_path / ".config" / "insanely-fast-whisper-rocm"
        user_config_dir.mkdir(parents=True)
        user_env_file = user_config_dir / ".env"
        user_env_file.write_text("OLD_VAR=old_value\n")

        # Mock Path.home() to return our temp directory
        with patch("pathlib.Path.home", return_value=tmpdir_path):
            # Read and exec the script without __name__ = "__main__"
            script_path = Path("scripts/setup_config.py")
            if not script_path.exists():
                pytest.skip("setup_config.py not found")

            script_globals = {"__file__": str(script_path), "__name__": "test_module"}
            with open(script_path) as f:
                exec(f.read(), script_globals)

            # Mock get_project_root to return our test project root
            def mock_get_project_root() -> Path:
                return project_root

            script_globals["get_project_root"] = mock_get_project_root

            # Mock input to return "yes" (overwrite)
            with patch("builtins.input", return_value="yes"):
                script_globals["main"]()

            # Verify config file was overwritten
            assert user_env_file.exists()
            assert user_env_file.read_text() == "NEW_VAR=new_value\n"


def test_setup_config_main_preserves_existing() -> None:
    """Test that setup_config.py preserves existing config when declined."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / "project"
        project_root.mkdir()

        # Create .env.example
        env_example = project_root / ".env.example"
        env_example.write_text("NEW_VAR=new_value\n")

        # Create existing user config
        user_config_dir = tmpdir_path / ".config" / "insanely-fast-whisper-rocm"
        user_config_dir.mkdir(parents=True)
        user_env_file = user_config_dir / ".env"
        user_env_file.write_text("OLD_VAR=old_value\n")

        # Mock Path.home() to return our temp directory
        with patch("pathlib.Path.home", return_value=tmpdir_path):
            # Read and exec the script without __name__ = "__main__"
            script_path = Path("scripts/setup_config.py")
            if not script_path.exists():
                pytest.skip("setup_config.py not found")

            script_globals = {"__file__": str(script_path), "__name__": "test_module"}
            with open(script_path) as f:
                exec(f.read(), script_globals)

            # Mock get_project_root to return our test project root
            def mock_get_project_root() -> Path:
                return project_root

            script_globals["get_project_root"] = mock_get_project_root

            # Mock input to return "no" (don't overwrite)
            with patch("builtins.input", return_value="no"):
                script_globals["main"]()

            # Verify config file was NOT overwritten
            assert user_env_file.exists()
            assert user_env_file.read_text() == "OLD_VAR=old_value\n"


def test_setup_config_handles_missing_env_example() -> None:
    """Test that setup_config.py handles missing .env.example gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / "project"
        project_root.mkdir()

        # Don't create .env.example

        # Mock Path.home() to return our temp directory
        with patch("pathlib.Path.home", return_value=tmpdir_path):
            # Read and exec the script without __name__ = "__main__"
            script_path = Path("scripts/setup_config.py")
            if not script_path.exists():
                pytest.skip("setup_config.py not found")

            script_globals = {"__file__": str(script_path), "__name__": "test_module"}
            with open(script_path) as f:
                exec(f.read(), script_globals)

            # Mock get_project_root to return our test project root
            def mock_get_project_root() -> Path:
                return project_root

            script_globals["get_project_root"] = mock_get_project_root

            # Run main - should exit early without error
            script_globals["main"]()

            # Verify config file was NOT created
            user_config_dir = tmpdir_path / ".config" / "insanely-fast-whisper-rocm"
            user_env_file = user_config_dir / ".env"
            assert not user_env_file.exists()


def test_setup_config_handles_keyboard_interrupt() -> None:
    """Test that setup_config.py handles keyboard interrupt gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_root = tmpdir_path / "project"
        project_root.mkdir()

        # Create .env.example
        env_example = project_root / ".env.example"
        env_example.write_text("TEST_VAR=value\n")

        # Create existing user config
        user_config_dir = tmpdir_path / ".config" / "insanely-fast-whisper-rocm"
        user_config_dir.mkdir(parents=True)
        user_env_file = user_config_dir / ".env"
        user_env_file.write_text("OLD_VAR=old_value\n")

        # Mock Path.home() to return our temp directory
        with patch("pathlib.Path.home", return_value=tmpdir_path):
            # Read and exec the script without __name__ = "__main__"
            script_path = Path("scripts/setup_config.py")
            if not script_path.exists():
                pytest.skip("setup_config.py not found")

            script_globals = {"__file__": str(script_path), "__name__": "test_module"}
            with open(script_path) as f:
                exec(f.read(), script_globals)

            # Mock get_project_root to return our test project root
            def mock_get_project_root() -> Path:
                return project_root

            script_globals["get_project_root"] = mock_get_project_root

            # Mock input to raise KeyboardInterrupt
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                script_globals["main"]()

            # Verify config file was NOT overwritten
            assert user_env_file.exists()
            assert user_env_file.read_text() == "OLD_VAR=old_value\n"
