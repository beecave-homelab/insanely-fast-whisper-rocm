"""Hugging Face model/cache manager CLI.

This script provides convenient commands to:
- pull (download) models
- list locally cached repos
- remove cached repos (all revisions)
- show/set/unset the Hugging Face cache dir used by this project (.env)
- show/set/unset the project's default download root (.env)

Notes:
- Changing environment variables via this CLI updates the repository `.env`.
  Open terminals/processes must be restarted to pick up changes.
- The effective Hugging Face cache location follows the documented precedence:
  HF_HUB_CACHE > HF_HOME/hub > XDG_CACHE_HOME/huggingface/hub >
  ~/.cache/huggingface/hub.

"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import typer
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, HFValidationError
from rich import box
from rich.console import Console
from rich.table import Table

# Typer app and groups
app = typer.Typer(help="Manage Hugging Face models and cache")
cache_app = typer.Typer(help="Show or modify the Hugging Face cache directory")
project_app = typer.Typer(help="Manage project-level defaults in .env")

# Rich console for pretty output
CONSOLE = Console()

# Project-specific env var for default download root used by this repo
ENV_PROJECT_DOWNLOAD_ROOT = "FWR_TRANSCRIBE_DOWNLOAD_ROOT"


# -----------
# Utilities
# -----------


def _repo_root() -> Path:
    """Return the repository root.

    Assumes this script resides in the `scripts/` directory and goes one level up.

    Returns:
        Path: Absolute path to the repository root.

    """
    return Path(__file__).resolve().parents[1]


def _env_file_path() -> Path:
    """Return the `.env` file path at the repository root.

    Returns:
        Path: Absolute path to `.env`.

    """
    return _repo_root() / ".env"


def _read_env_lines() -> list[str]:
    """Read the `.env` file as a list of lines.

    Returns:
        list[str]: Lines of the `.env` file or an empty list if not present.

    """
    env_path = _env_file_path()
    if not env_path.exists():
        return []
    return env_path.read_text(encoding="utf-8").splitlines()


def _write_env_lines(lines: Iterable[str]) -> None:
    """Write the given lines back to the `.env` file.

    Args:
        lines (Iterable[str]): Lines to write.

    """
    content = "\n".join(lines) + "\n" if lines else ""
    _env_file_path().write_text(content, encoding="utf-8")


def _set_env_var_in_dotenv(key: str, value: str | None) -> None:
    """Create/update/remove a key in the project's `.env` file.

    If ``value`` is ``None``, the key is removed. Otherwise it's set.

    Args:
        key (str): Environment variable name.
        value (Optional[str]): Value to set or ``None`` to remove.

    """
    lines = _read_env_lines()

    new_lines: list[str] = []
    found = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if stripped.startswith(f"{key}="):
            if value is not None:
                # Always quote to be safe for paths with spaces
                new_lines.append(f'{key}="{value}"')
            # If value is None, skip (remove key)
            found = True
        else:
            new_lines.append(line)

    if not found and value is not None:
        new_lines.append(f'{key}="{value}"')

    _write_env_lines(new_lines)


# -------------------------
# Hugging Face cache utils
# -------------------------


def _effective_hf_cache_dir() -> Path:
    """Compute the effective Hugging Face cache directory.

    Precedence order:
    1. ``HF_HUB_CACHE``
    2. ``HF_HOME`` + ``/hub``
    3. ``XDG_CACHE_HOME`` + ``/huggingface/hub``
    4. ``~/.cache/huggingface/hub``

    Returns:
        Path: Resolved path to the effective cache directory.

    """
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser().resolve()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve() / "hub"

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser().resolve() / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _format_ts(ts: float) -> str:
    """Format a POSIX timestamp for display.

    Args:
        ts (float): POSIX timestamp.

    Returns:
        str: Human-readable time or ``"-"`` if invalid.

    """
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _latest_snapshot_dir(repo_path: Path) -> Path | None:
    """Return the most recently modified snapshot directory for a repo.

    Args:
        repo_path (Path): Base path of the repo in the HF cache.

    Returns:
        Path | None: Path to the latest ``snapshots/<rev>`` directory or ``None``.

    """
    snapshots = repo_path / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    except Exception:
        return None
    if not dirs:
        return None
    # Pick by modification time (best-effort heuristic)
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _detect_model_framework(snapshot_dir: Path | None) -> str:
    """Best-effort detection of model framework based on files present.

    Args:
        snapshot_dir (Path | None): A snapshot directory to inspect.

    Returns:
        str: One of ``pytorch``, ``tensorflow``, ``flax``, ``onnx``,
            ``ctranslate2``, or ``unknown``.

    """
    if snapshot_dir is None or not snapshot_dir.exists():
        return "unknown"
    try:
        names = {p.name for p in snapshot_dir.iterdir() if p.is_file()}
    except Exception:
        names = set()

    # PyTorch
    if (
        "pytorch_model.bin" in names
        or any(n.endswith(".safetensors") for n in names)
        or "model.safetensors" in names
    ):
        return "pytorch"

    # TensorFlow
    if "tf_model.h5" in names:
        return "tensorflow"

    # Flax
    if "flax_model.msgpack" in names:
        return "flax"

    # ONNX
    if any(n.endswith(".onnx") for n in names):
        return "onnx"

    # CTranslate2 (heuristics):
    # - presence of model.bin (not a PyTorch shard name)
    # - plus an index json or known vocab files frequently shipped with CT2
    if "model.bin" in names:
        if (
            "model.bin.index.json" in names
            or any(n.startswith("model.bin.") and n.endswith(".json") for n in names)
            or any(
                ("vocabulary" in n and n.endswith(".txt"))
                or n.endswith("sentencepiece.bpe")
                or n.endswith("sentencepiece.model")
                for n in names
            )
        ):
            return "ctranslate2"

    return "unknown"


def _style_repo_type(repo_type: str) -> str:
    """Return a Rich-styled string for the repository type.

    Args:
        repo_type (str): Repository type (e.g., "model", "dataset", "space").

    Returns:
        str: Rich markup string with color applied to the repo type.

    """
    color_map: dict[str, str] = {
        "model": "cyan",
        "dataset": "magenta",
        "space": "blue",
    }
    color = color_map.get(repo_type, "white")
    return f"[{color}]{repo_type}[/{color}]"


def _style_framework(framework: str) -> str:
    """Return a Rich-styled string for the detected model framework.

    Args:
        framework (str): Detected framework (e.g., "pytorch", "tensorflow").

    Returns:
        str: Rich markup string with color applied to the framework name.

    """
    color_map: dict[str, str] = {
        "pytorch": "red",
        "tensorflow": "yellow",
        "flax": "green",
        "onnx": "bright_cyan",
        "ctranslate2": "bright_magenta",
        "unknown": "dim",
    }
    color = color_map.get(framework, "white")
    if framework == "unknown":
        return "[dim]unknown[/dim]"
    return f"[{color}]{framework}[/{color}]"


# -------
# pull
# -------


@app.command("pull")
def pull(
    repo_id: str = typer.Argument(
        ..., help="Repository id, e.g. 'openai/whisper-tiny'"
    ),
    revision: str | None = typer.Option(
        None, help="Specific revision (branch, tag or commit hash)"
    ),
    cache_dir: Path | None = typer.Option(
        None,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help=(
            "Override cache dir for this command only. Defaults to effective HF cache."
        ),
    ),
    local_files_only: bool = typer.Option(
        False, help="Do not download; require files to be present locally"
    ),
    allow: list[str] = typer.Option(
        None, "--allow", help="Glob patterns to include (repeat for multiple)"
    ),
    ignore: list[str] = typer.Option(
        None, "--ignore", help="Glob patterns to exclude (repeat for multiple)"
    ),
    token: str | None = typer.Option(
        None, help="HF token (otherwise env or keyring is used)"
    ),
    force: bool = typer.Option(
        False, help="Force re-download (ignored with --local-files-only)"
    ),
) -> None:
    """Download a model repository snapshot if needed.

    Prints the resolved snapshot path on success. Exits non-zero on failure.

    Args:
        repo_id (str): Repository id (e.g., ``"openai/whisper-tiny"``).
        revision (Optional[str]): Specific revision (branch, tag or commit).
        cache_dir (Optional[Path]): Cache directory override for this command.
        local_files_only (bool): If True, do not access the network.
        allow (list[str]): Glob patterns to include.
        ignore (list[str]): Glob patterns to exclude.
        token (Optional[str]): Hugging Face token to authenticate.
        force (bool): Force re-download (ignored with ``--local-files-only``).

    Raises:
        typer.Exit: If any validation or HTTP error occurs.

    """
    try:
        resolved_cache = cache_dir or _effective_hf_cache_dir()
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=resolved_cache,
            local_files_only=local_files_only,
            allow_patterns=allow if allow else None,
            ignore_patterns=ignore if ignore else None,
            token=token,
            force_download=force and not local_files_only,
        )
        typer.echo(path)
    except HFValidationError as e:
        typer.secho(f"Invalid repo id or args: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e
    except HfHubHTTPError as e:
        typer.secho(f"Hub HTTP error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=3) from e
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from e
    except Exception as e:  # pragma: no cover - safety net
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=5) from e


# -------
# list
# -------


@app.command("list")
def list_cached(
    cache_dir: Path | None = typer.Option(
        None,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=False,
        help="Scan a specific cache dir instead of the effective one",
    ),
    repo_type: str | None = typer.Option(
        "model", help="Filter by repo type: model|dataset|space|all"
    ),
    contains: str | None = typer.Option(
        None, help="Filter repos by substring in repo_id"
    ),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    less: bool = typer.Option(
        False, "--less", help="Show minimal columns for a compact view"
    ),
    more: bool = typer.Option(
        False, "--more", help="Show extended details (all columns)"
    ),
) -> None:
    """List cached repositories in the HF cache directory.

    Args:
        cache_dir (Optional[Path]): Target cache directory.
        repo_type (Optional[str]): Filter by repo type or 'all'.
        contains (Optional[str]): Substring filter on repo id.
        json_out (bool): If True, prints JSON instead of text.
        less (bool): If True, prints a minimal set of columns.
        more (bool): If True, prints all available columns (extended).

    Raises:
        typer.Exit: If no cached repositories are found (exit code 0) or when
            invalid flag combinations are used.

    """
    info = scan_cache_dir(cache_dir=cache_dir or _effective_hf_cache_dir())

    def _type_ok(t: str) -> bool:
        return repo_type in (None, "all") or t == repo_type

    items = []
    for repo in sorted(info.repos, key=lambda r: r.repo_id):
        if not _type_ok(repo.repo_type):
            continue
        if contains and contains not in repo.repo_id:
            continue
        snapshot_dir = _latest_snapshot_dir(repo.repo_path)
        framework = _detect_model_framework(snapshot_dir)
        item = {
            "repo_id": repo.repo_id,
            "type": repo.repo_type,
            "framework": framework,
            "size": repo.size_on_disk_str,  # human readable
            "nb_files": repo.nb_files,
            "last_accessed": _format_ts(repo.last_accessed),
            "last_modified": _format_ts(repo.last_modified),
            "revisions": sorted(r.commit_hash for r in repo.revisions),
            "path": str(repo.repo_path),
        }
        items.append(item)

    if json_out:
        typer.echo(json.dumps(items, indent=2))
    else:
        if not items:
            CONSOLE.print("[yellow]No cached repositories found.[/yellow]")
            raise typer.Exit(code=0)

        if less and more:
            typer.secho(
                "Use only one of --less or --more.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        # Determine view mode
        view_more = bool(more)
        view_less = bool(less)

        # Build table according to selected mode
        table = Table(
            title="Hugging Face Cache",
            box=box.SQUARE,
            expand=True,
            header_style="bold",
        )

        if view_more:
            # Extended view: all columns (original detailed view)
            table.add_column("repo_id", style="bold", overflow="fold")
            table.add_column("type")
            table.add_column("framework")
            table.add_column("size", justify="right")
            table.add_column("files", justify="right")
            table.add_column("last_accessed")
            table.add_column("last_modified")
            table.add_column("path", overflow="fold")
            table.add_column("revisions", overflow="fold")

            for it in items:
                revs_str = ", ".join(it["revisions"]) if it["revisions"] else "-"
                table.add_row(
                    it["repo_id"],
                    _style_repo_type(it["type"]),
                    _style_framework(it["framework"]),
                    it["size"],
                    str(it["nb_files"]),
                    it["last_accessed"],
                    it["last_modified"],
                    f"[dim]{it['path']}[/dim]",
                    f"[dim]{revs_str}[/dim]",
                )
        elif view_less:
            # Minimal view: repo_id, type, size
            table.add_column("repo_id", style="bold", overflow="fold")
            table.add_column("type")
            table.add_column("size", justify="right")

            for it in items:
                table.add_row(
                    it["repo_id"],
                    _style_repo_type(it["type"]),
                    it["size"],
                )
        else:
            # Default compact view: fewer columns than full
            table.add_column("repo_id", style="bold", overflow="fold")
            table.add_column("type")
            table.add_column("framework")
            table.add_column("size", justify="right")
            table.add_column("files", justify="right")
            table.add_column("last_accessed")

            for it in items:
                table.add_row(
                    it["repo_id"],
                    _style_repo_type(it["type"]),
                    _style_framework(it["framework"]),
                    it["size"],
                    str(it["nb_files"]),
                    it["last_accessed"],
                )

        CONSOLE.print(table)


# ----------
# remove
# ----------


@app.command("remove")
def remove_cached(
    repo_id: str = typer.Argument(
        ..., help="Repo id to remove from cache (all revisions)"
    ),
    cache_dir: Path | None = typer.Option(
        None,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Target cache directory",
    ),
    repo_type: str = typer.Option("model", help="Type of repo: model|dataset|space"),
    dry_run: bool = typer.Option(
        False, help="Compute expected freed size without deleting"
    ),
    yes: bool = typer.Option(False, "-y", help="Do not prompt for confirmation"),
) -> None:
    """Remove a cached repository (all cached revisions).

    Args:
        repo_id (str): Repository id to remove.
        cache_dir (Optional[Path]): Target cache directory.
        repo_type (str): Repo type (model|dataset|space).
        dry_run (bool): Only compute expected freed size.
        yes (bool): Skip confirmation prompt.

    Raises:
        typer.Exit: If user cancels the operation.

    """
    info = scan_cache_dir(cache_dir=cache_dir or _effective_hf_cache_dir())

    target_hashes: list[str] = []
    target_paths: set[Path] = set()

    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == repo_type:
            target_hashes.extend(r.commit_hash for r in repo.revisions)
            target_paths.add(repo.repo_path)

    if not target_hashes:
        typer.secho("No matching cached repo found.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    strategy = info.delete_revisions(*target_hashes)

    if dry_run:
        typer.echo(f"Would free {strategy.expected_freed_size_str} by deleting:")
        for p in sorted(strategy.repos):
            typer.echo(f" - {p}")
        raise typer.Exit(code=0)

    if not yes:
        confirm = typer.confirm(
            f"Delete {len(strategy.repos)} repo(s) and "
            f"{len(strategy.snapshots)} snapshot(s) (free "
            f"{strategy.expected_freed_size_str})?",
            default=False,
        )
        if not confirm:
            typer.echo("Cancelled.")
            raise typer.Exit(code=0)

    strategy.execute()
    typer.echo(
        f"Deleted {len(strategy.repos)} repo(s), freed "
        f"{strategy.expected_freed_size_str}."
    )


# -------
# cleanup
# -------


@app.command("cleanup")
def cleanup(
    days: int = typer.Argument(
        ..., help="Delete repos not accessed in the last N days"
    ),
    cache_dir: Path | None = typer.Option(
        None,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Target cache directory",
    ),
    repo_type: str | None = typer.Option(
        "model", help="Filter by repo type: model|dataset|space|all"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--no-dry-run", help="Preview deletions without executing"
    ),
) -> None:
    """Delete cached repos not accessed for the given number of days.

    This command evaluates each cached repository's last access time and marks it
    for deletion if it hasn't been accessed for at least ``days``. When not in
    dry-run mode, it asks for confirmation for each repo before deletion.

    Args:
        days (int): Threshold in days since last access.
        cache_dir (Path | None): Cache directory to operate on.
        repo_type (str | None): Filter by repo type or 'all'.
        dry_run (bool): If True, only shows what would be deleted.

    Raises:
        typer.Exit: If no candidates are found or operation is cancelled.

    """
    cutoff_ts = datetime.now().timestamp() - (days * 24 * 60 * 60)
    info = scan_cache_dir(cache_dir=cache_dir or _effective_hf_cache_dir())

    def _type_ok(t: str) -> bool:
        return repo_type in (None, "all") or t == repo_type

    # Select candidate repos
    candidates = []
    for repo in info.repos:
        if not _type_ok(repo.repo_type):
            continue
        last_ts = repo.last_accessed or repo.last_modified or 0.0
        if last_ts <= cutoff_ts:
            candidates.append(repo)

    if not candidates:
        typer.echo("No repositories eligible for cleanup.")
        raise typer.Exit(code=0)

    # Sort by last access (oldest first)
    candidates.sort(key=lambda r: r.last_accessed or r.last_modified or 0.0)

    # Prepare pretty table for dry-run summary
    table: Table | None = None
    if dry_run:
        table = Table(
            title="[DRY-RUN] Candidates for deletion",
            box=box.SQUARE,
            expand=True,
            header_style="bold",
        )
        table.add_column("repo_id", style="bold", overflow="fold")
        table.add_column("type")
        table.add_column("size", justify="right")
        table.add_column("last_accessed")
        table.add_column("last_modified")
        table.add_column("path", overflow="fold")
        table.add_column("expected_freed", justify="right")

    for repo in candidates:
        rev_hashes = [r.commit_hash for r in repo.revisions]
        if not rev_hashes:
            # Nothing tangible to delete for this repo
            continue
        strategy = info.delete_revisions(*rev_hashes)

        if dry_run and table is not None:
            table.add_row(
                repo.repo_id,
                _style_repo_type(repo.repo_type),
                repo.size_on_disk_str,
                _format_ts(repo.last_accessed),
                _format_ts(repo.last_modified),
                f"[dim]{repo.repo_path}[/dim]",
                f"[green]{strategy.expected_freed_size_str}[/green]",
            )
            continue

        repo_header = (
            f"{repo.repo_id}  (type: {repo.repo_type}, size: {repo.size_on_disk_str})"
        )
        repo_meta = (
            f"  last_accessed: {_format_ts(repo.last_accessed)}\n"
            f"  last_modified: {_format_ts(repo.last_modified)}\n"
            f"  path: {repo.repo_path}"
        )

        # Confirm per repo
        typer.echo(repo_header)
        typer.echo(repo_meta)
        confirm = typer.confirm("Delete this repo from cache?", default=False)
        if not confirm:
            typer.echo("Skipped.\n")
            continue

        strategy.execute()
        CONSOLE.print(
            f"[green]Deleted.[/green] Freed approximately "
            f"{strategy.expected_freed_size_str}.\n"
        )

    if dry_run and table is not None:
        CONSOLE.print(table)
        return


# ----------------
# cache-dir group
# ----------------


@cache_app.command("show")
def cache_show() -> None:
    """Print the effective Hugging Face cache directory path."""
    typer.echo(str(_effective_hf_cache_dir()))


@cache_app.command("set")
def cache_set(
    path: Path = typer.Argument(..., help="Directory to use for HF_HUB_CACHE"),
    create_dir: bool = typer.Option(
        True, help="Create the directory if it doesn't exist"
    ),
) -> None:
    """Set ``HF_HUB_CACHE`` in the project's `.env` to the given path.

    Args:
        path (Path): Directory to use as Hugging Face cache.
        create_dir (bool): Create directory if missing.

    """
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    _set_env_var_in_dotenv("HF_HUB_CACHE", str(path))
    typer.echo(f"Set HF_HUB_CACHE to: {path}")


@cache_app.command("unset")
def cache_unset() -> None:
    """Unset ``HF_HUB_CACHE`` from the project's `.env`.

    Reverts to the default cache resolution strategy.
    """
    _set_env_var_in_dotenv("HF_HUB_CACHE", None)
    typer.echo("Unset HF_HUB_CACHE (will use default resolution)")


# ---------------------------
# project defaults (.env) grp
# ---------------------------


@project_app.command("show-download-root")
def project_show_download_root() -> None:
    """Print the project's default download root.

    Reads from the persisted `.env` if present, otherwise from the environment.
    Prints ``-`` if not set.
    """
    val = os.getenv(ENV_PROJECT_DOWNLOAD_ROOT)  # value for current process
    # Prefer reading from file to show persisted value
    persisted = None
    for line in _read_env_lines():
        stripped = line.strip()
        if stripped.startswith(f"{ENV_PROJECT_DOWNLOAD_ROOT}="):
            persisted = stripped.split("=", 1)[1].strip().strip('"')
            break
    typer.echo(persisted or val or "-")


@project_app.command("set-download-root")
def project_set_download_root(
    path: Path = typer.Argument(
        ..., help="Directory to use for project default download root"
    ),
    create_dir: bool = typer.Option(
        True, help="Create the directory if it doesn't exist"
    ),
) -> None:
    """Set the project's default download root in `.env`.

    Args:
        path (Path): Directory path to set.
        create_dir (bool): Create directory if missing.

    """
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    _set_env_var_in_dotenv(ENV_PROJECT_DOWNLOAD_ROOT, str(path))
    typer.echo(f"Set {ENV_PROJECT_DOWNLOAD_ROOT} to: {path}")


@project_app.command("unset-download-root")
def project_unset_download_root() -> None:
    """Unset the project's default download root from `.env`."""
    _set_env_var_in_dotenv(ENV_PROJECT_DOWNLOAD_ROOT, None)
    typer.echo(f"Unset {ENV_PROJECT_DOWNLOAD_ROOT}")


# Register groups
app.add_typer(cache_app, name="cache-dir")
app.add_typer(project_app, name="project")


def _main() -> None:
    """Entry-point for CLI execution."""
    app()


if __name__ == "__main__":
    _main()
