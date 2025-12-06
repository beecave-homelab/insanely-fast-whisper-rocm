"""Test utilities shared across API test modules."""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@lru_cache(maxsize=1)
def get_project_version() -> str:
    """Return the project version declared in package metadata or `pyproject.toml`.

    Returns:
        str: Semantic version string for the project distribution.

    Raises:
        RuntimeError: If the version cannot be read from metadata or `pyproject.toml`.
    """
    try:
        return metadata.version("insanely-fast-whisper-rocm")
    except metadata.PackageNotFoundError:
        config = Path(__file__).resolve().parents[1] / "pyproject.toml"
        raw = config.read_text(encoding="utf-8")
        if tomllib is not None:
            data = tomllib.loads(raw)
            project = data.get("project", {})
            version = project.get("version")
        else:
            version = _extract_version_from_text(raw)
        if not isinstance(version, str) or not version:
            msg = "Could not determine project version from pyproject.toml"
            raise RuntimeError(msg)
        return version


def _extract_version_from_text(text: str) -> str | None:
    """Extract the project version from `pyproject.toml` content.

    Args:
        text: Raw `pyproject.toml` contents.

    Returns:
        str | None: The version string if it can be found.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            _, value = stripped.split("=", 1)
            candidate = value.strip().strip('"')
            if candidate:
                return candidate
    return None
