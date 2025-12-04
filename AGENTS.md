# AGENTS.md — Coding Rules (Ruff + Pytest + SOLID)

This repository uses **Ruff** as the single source of truth for linting/formatting and **Pytest** (with **pytest-cov**) for tests & coverage. CI fails when these rules are violated.

Run locally before committing:

```bash
# Lint & format (Ruff)
pdm run ruff check --fix .
pdm run ruff format .

# Tests & coverage (adjust --cov target if needed)
pdm run pytest --maxfail=1 -q
pdm run pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml
```

When in doubt, prefer **correctness → clarity → consistency → brevity** (in that order).

## Table of Contents

- [1) Correctness (Ruff F - Pyflakes)](#1-correctness-ruff-f---pyflakes)
- [2) PEP 8 surface rules (Ruff E, W - pycodestyle)](#2-pep-8-surface-rules-ruff-e-w---pycodestyle)
- [3) Naming conventions (Ruff N - pep8-naming)](#3-naming-conventions-ruff-n---pep8-naming)
- [4) Imports: order & style (Ruff I - isort rules)](#4-imports-order--style-ruff-i---isort-rules)
- [5) Docstrings — content & style (Ruff D + DOC)](#5-docstrings--content--style-ruff-d--doc)
- [6) Import hygiene (Ruff TID - flake8-tidy-imports)](#6-import-hygiene-ruff-tid---flake8-tidy-imports)
- [7) Modern Python upgrades (Ruff UP - pyupgrade)](#7-modern-python-upgrades-ruff-up---pyupgrade)
- [8) Future annotations (Ruff FA - flake8-future-annotations)](#8-future-annotations-ruff-fa---flake8-future-annotations)
- [9) Local ignores (only when justified)](#9-local-ignores-only-when-justified)
- [10) Tests & examples (Pytest + Coverage)](#10-tests--examples-pytest--coverage)
- [11) Commit discipline](#11-commit-discipline)
- [12) Quick DO / DON’T](#12-quick-do--dont)
- [13) Pre-commit (recommended)](#13-pre-commit-recommended)
- [14) CI expectations](#14-ci-expectations)
- [15) SOLID design principles — Explanation & Integration](#15-solid-design-principles--explanation--integration)
- [16) Configuration management — environment variables & constants](#16-configuration-management--environment-variables--constants)
- [Final note](#final-note)

---

## 1) Correctness (Ruff F - Pyflakes)

### What It Enforces — Correctness

- No undefined names/variables.
- No unused imports/variables/arguments.
- No duplicate arguments in function definitions.
- No `import *`.

### Agent Checklist — Correctness

- Remove dead code and unused symbols.
- Keep imports minimal and explicit.
- Use local scopes (comprehensions, context managers) where appropriate.
- Do **not** read configuration from `os.environ` directly outside the dedicated constants module (see section 16).

---

## 2) PEP 8 surface rules (Ruff E, W - pycodestyle)

### What It Enforces — PEP 8 Surface

- Spacing/blank-line/indentation hygiene.
- No trailing whitespace.
- Reasonable line breaks; respect the configured line length (see `pyproject.toml` or `ruff.toml`).

### Agent Checklist — PEP 8 Surface

- Let the formatter handle whitespace.
- Break long expressions cleanly (after operators, around commas).
- End files with exactly one trailing newline.

---

## 3) Naming conventions (Ruff N - pep8-naming)

### What It Enforces — Naming

- `snake_case` for functions, methods, and non-constant variables.
- `CapWords` (PascalCase) for classes.
- `UPPER_CASE` for module-level constants.
- Exceptions end with `Error` and subclass `Exception`.

### Agent Checklist — Naming

- Avoid camelCase unless mirroring a third-party API; if unavoidable, use a targeted pragma for that line only.

---

## 4) Imports: order & style (Ruff I - isort rules)

### What It Enforces — Imports

- Group imports: 1) Standard library, 2) Third-party, 3) First-party/local.
- Alphabetical within groups; one blank line between groups.
- Prefer one import per line for clarity.

### Agent Checklist — Imports

- Keep imports at module scope (top of file).
- Only alias when it adds clarity (e.g., `import numpy as np`).

### Canonical example — Imports

```python
from __future__ import annotations

import dataclasses
import pathlib

import httpx
import pydantic

from yourpkg.core import config
from yourpkg.utils.paths import ensure_dir
```

*(Replace `yourpkg` with your top-level package. In app-only repos, keep first-party imports minimal.)*

---

## 5) Docstrings — content & style (Ruff D + DOC)

Public modules, classes, functions, and methods **must have docstrings**. Ruff enforces **pydocstyle** (`D…`) and **pydoclint** (`DOC…`).

**Single-source style**: **Google-style** docstrings with type hints in signatures.

### Rules of Thumb — Docstrings

- Triple double quotes.
- First line: one-sentence summary, capitalized, ends with a period.
- Blank line after summary, then details.
- Keep `Args/Returns/Raises` in sync with the signature.
- Use imperative mood (“Return…”, “Validate…”). Don’t repeat obvious types (use type hints).

### Function Template — Docstrings

```python
def frobnicate(path: pathlib.Path, *, force: bool = False) -> str:
    """Frobnicate the resource at ``path``.

    Performs an idempotent frobnication. If ``force`` is true, existing
    artifacts will be replaced.

    Args:
        path: Filesystem location of the target resource.
        force: Replace previously generated artifacts if present.

    Returns:
        A stable identifier for the frobnicated resource.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        PermissionError: If write access is denied.
    """
```

### Class Template — Docstrings

```python
class ResourceManager:
    """Coordinate creation and lifecycle of resources.

    Notes:
        Thread-safe for read operations; writes are serialized.
    """
```

---

## 6) Import hygiene (Ruff TID - flake8-tidy-imports)

### What It Enforces — Import Hygiene

- Prefer absolute imports over deep relative imports.
- Avoid circular imports; import inside functions only for performance or to break a cycle.
- Avoid broad implicit re-exports; if you re-export, do it explicitly via `__all__`.

### Agent Checklist — Import Hygiene

```python
try:
    import rich
except ModuleNotFoundError:  # pragma: no cover
    rich = None  # type: ignore[assignment]
```

---

## 7) Modern Python upgrades (Ruff UP - pyupgrade)

### What It Prefers — Modernization

- f-strings over `format()` / `%`.
- PEP 585 generics (`list[str]`, `dict[str, int]`) over `typing.List`, `typing.Dict`, etc.
- Context managers where appropriate.
- Remove legacy constructs (`six`, `u''`, redundant `object`).

### Agent Checklist — Modernization

- Use `pathlib.Path` for filesystem paths.
- Use assignment expressions (`:=`) sparingly and only when clearer.
- Prefer `is None`/`is not None`.

---

## 8) Future annotations (Ruff FA - flake8-future-annotations)

### Guidance — Future Annotations

- Targeting **Python < 3.11**: place at the top of every module:

  ```python
  from __future__ import annotations
  ```

- Targeting **Python ≥ 3.11**: you may omit it; align the `FA` rule in Ruff config.

---

## 9) Local ignores (only when justified)

### Policy — Local Ignores

Prefer fixing the root cause. If a one-off ignore is necessary, keep it **scoped and documented**:

```python
value = compute()  # noqa: F401  # used by plugin loader via reflection
```

For docstring mismatches caused by third-party constraints, use a targeted `# noqa: D…, DOC…` with a brief reason.

---

## 10) Tests & examples (Pytest + Coverage)

### Expectations — Tests

- Tests follow the same rules as production code.
- Naming: `test_<unit_under_test>__<expected_behavior>()`.
- Keep tests deterministic; avoid hidden network/filesystem dependencies without fixtures.

### Minimal Example — Tests

```python
def add(a: int, b: int) -> int:
    """Return the sum of two integers.

    Examples:
        >>> add(2, 3)
        5
    """
```

### Running — Tests & Coverage

```bash
# Quick
pdm run pytest -q

# Coverage (adjust --cov target to your package or ".")
pdm run pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml
```

### Coverage Policy — Threshold

- Guideline: **≥ 85%** line coverage, with critical paths covered.
- Make CI fail below the threshold (see “CI expectations”).

---

## 11) Commit discipline

### Expectations — Commits

Run Ruff and tests **before** committing. Keep commits small and focused.

Use your project’s conventional commit format.

---

## 12) Quick DO / DON’T

### DO — Practices

- Google-style docstrings that match signatures.
- Absolute imports and sorted import blocks.
- f-strings and modern type syntax (`list[str]`).
- Remove unused code promptly.
- Use Pytest fixtures for reusable setup; prefer `tmp_path` for temp files.

### DON’T — Anti-patterns

- Introduce camelCase (except when mirroring external APIs).
- Use `import *` or deep relative imports.
- Leave parameters undocumented in public functions.
- Add broad `noqa`—always keep ignores narrow and justified.

---

## 13) Pre-commit (recommended)

### Configuration — Pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9  # keep in sync with your chosen Ruff version
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## 14) CI expectations

### Commands — CI

```bash
# Lint & format
pdm run ruff check .
pdm run ruff format --check .

# Tests & coverage
pdm run pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml --maxfail=1
```

### Policy — CI Coverage

Enforce a minimum coverage threshold (example: 85%). Fail the pipeline if below.

---

## 15) SOLID design principles — Explanation & Integration

The **SOLID** principles help you design maintainable, testable, and extensible Python code. This section explains each principle concisely and shows how it maps to our linting, docs, and tests.

### S — Single Responsibility Principle (SRP)

- **Definition**: A module/class should have **one reason to change** (one cohesive responsibility).
- **Pythonic approach**:
  - Keep classes small; factor out I/O, parsing, and domain logic into distinct units.
  - Prefer composition over “god classes”.
- **In practice**:
  - Split functions that both “validate & write to disk” into separate units.
  - Move side-effects (I/O, network) behind narrow interfaces.
- **How we enforce/integrate**:
  - **Docs**: Each public class/function docstring states its single responsibility.
  - **Tests**: Unit tests focus on one behavior per test (narrow fixtures).
  - **Lint**: Large files/functions are a smell (consider refactor even if Ruff passes).

### O — Open/Closed Principle (OCP)

- **Definition**: Software entities should be **open for extension, closed for modification**.
- **Pythonic approach**:
  - Rely on **polymorphism** via abstract base classes or `typing.Protocol`.
  - Inject strategies or policies instead of hard-coding conditionals.
- **In practice**:
  - Define `Storage` protocol with `write()` and implement `FileStorage`, `S3Storage` without changing callers.
- **How we enforce/integrate**:
  - **Docs**: Document stable extension points (interfaces/protocols) in module/class docstrings.
  - **Tests**: Parametrize tests across multiple implementations to validate substitutability.
  - **Lint**: Keep imports clean; avoid “if type == …” switches in hot paths.

### L — Liskov Substitution Principle (LSP)

- **Definition**: Subtypes must be **substitutable** for their base types without breaking expectations.
- **Pythonic approach**:
  - Subclasses must not strengthen preconditions or weaken postconditions.
  - Keep method signatures compatible (types/return values/raised errors).
- **In practice**:
  - If base `Repository.get(id) -> Model | None`, a subtype must not start raising on “not found”.
- **How we enforce/integrate**:
  - **Docs**: State behavioral contracts and possible exceptions in docstrings.
  - **Tests**: Run the same behavior tests against base and derived implementations (parametrized).
  - **Lint**: Ruff won’t prove LSP, but naming and import rules reduce confusion; rely on tests/contracts.

### I — Interface Segregation Principle (ISP)

- **Definition**: Prefer **small, role-specific interfaces** over fat interfaces.
- **Pythonic approach**:
  - Use multiple `Protocol`s (or ABCs) with narrowly scoped methods.
  - Accept only what you need at call sites (e.g., `Readable` protocol, not `FileLikeAndNetworkAndCache`).
- **In practice**:
  - Split `DataStore` into `Readable` and `Writable` where consumers only need one.
- **How we enforce/integrate**:
  - **Docs**: Clarify the minimal interface needed by a function/class (in the Args section).
  - **Tests**: Provide tiny fakes/mocks that implement just the required protocol.
  - **Lint**: Keep imports modular; avoid cyclic dependencies driven by bloated interfaces.

### D — Dependency Inversion Principle (DIP)

- **Definition**: High-level modules **depend on abstractions**, not concrete details.
- **Pythonic approach**:
  - Use constructor or function **dependency injection** of protocols/ABCs.
  - Keep wiring in a thin composition/bootstrap layer.
- **In practice**:
  - Class accepts `Clock` protocol; production uses `SystemClock`, tests pass `FrozenClock`.
- **How we enforce/integrate**:
  - **Docs**: Document injected dependencies and their contracts.
  - **Tests**: Replace dependencies with fakes/stubs; no slow/global state in unit tests.
  - **Lint**: Absolute imports and clean layering reduce unintended tight coupling.

### SOLID — Minimal example (Protocols + DI)

```python
from __future__ import annotations
from typing import Protocol
import pathlib

class Storage(Protocol):
    def write(self, path: pathlib.Path, data: bytes) -> None: ...

class FileStorage:
    def write(self, path: pathlib.Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

class Uploader:
    """Upload artifacts using an injected Storage (DIP, OCP, ISP).

    Args:
        storage: Minimal interface that supports 'write'.
    """
    def __init__(self, storage: Storage) -> None:
        self._storage = storage  # DIP

    def publish(self, dest: pathlib.Path, payload: bytes) -> None:
        # SRP: only orchestrates publication; no direct filesystem logic here.
        self._storage.write(dest, payload)

# LSP test idea: any Storage conformer can be used transparently (FakeStorage, S3Storage, ...).
```

### SOLID — Agent Checklist

- **SRP**: One responsibility per module/class; split I/O from domain logic.
- **OCP**: Use protocols/ABCs and strategy injection to extend without edits.
- **LSP**: Keep subtype behavior/contract compatible; parametrize tests over implementations.
- **ISP**: Prefer small protocols; accept only what you need.
- **DIP**: Depend on abstractions; inject dependencies (avoid hard-coded singletons/globals).

---

## 16) Configuration management — environment variables & constants

These rules standardize how environment variables are loaded and accessed across the codebase. They prevent config sprawl, enable testing, and align with **SRP** and **DIP**.

### 16.1 Single loading point

- Environment variables are parsed **exactly once** at application start.
- The loader function is `load_project_env()` located at `<package>/utils/env_loader.py`.

### 16.2 Central import location

- `load_project_env()` **MUST** be invoked **only** inside `<package>/utils/constant.py`.
- No other file may import `env_loader` or call `load_project_env()` directly.

### 16.3 Constant exposure

- After loading, `<package>/utils/constant.py` exposes project-wide configuration constants (e.g., `DEFAULT_CHUNK_LEN_SEC`, `DEFAULT_BATCH_SIZE`).
- All other modules (e.g., `<package>/app.py`, `<package>/transcribe.py`) **must import from** `<package>.utils.constant` instead of reading `os.environ` or `.env`.

### 16.4 Adding new variables

- Define a sensible default in `<package>/utils/constant.py` using `os.getenv("VAR_NAME", "default")` or typed parsing logic.
- Document every variable in `.env.example` with a short description and default.

### 16.5 Enforcement policy

- Pull requests that add direct `os.environ[...]` access or import `env_loader` outside `utils/constant.py` **must be rejected**.
- Suggested CI guardrail (example grep check):

  ```bash
  # deny direct env reads outside constants module
  ! git grep -nE 'os\\.environ\\[|os\\.getenv\\(' -- ':!<package>/utils/constant.py' ':!**/tests/**'
  ```

### 16.6 Example layout (illustrative)

```python
# <package>/utils/env_loader.py
from __future__ import annotations
import os

def load_project_env() -> dict[str, str]:
    # Parse once: could expand to load .env, validate, coerce types, etc.
    return dict(os.environ)  # Keep simple; real code may normalize keys/types
```

```python
# <package>/utils/constant.py
from __future__ import annotations
import os
from .env_loader import load_project_env

# Load once (single source of truth)
_ENV = load_project_env()

# Exposed constants (typed, with sensible defaults)
DEFAULT_CHUNK_LEN_SEC: int = int(_ENV.get("DEFAULT_CHUNK_LEN_SEC", "30"))
DEFAULT_BATCH_SIZE: int = int(_ENV.get("DEFAULT_BATCH_SIZE", "8"))
APP_ENV: str = _ENV.get("APP_ENV", "development")
```

```python
# <package>/app.py  (or any other module)
from __future__ import annotations
from <package>.utils.constant import DEFAULT_BATCH_SIZE

def run() -> None:
    # Use constants; do not read os.environ here
    ...
```

### 16.7 Testing guidance for configuration

- Unit tests may override constants via monkeypatching the **constants module attributes**, not the environment loader:

  ```python
  def test_behavior_with_small_batch(monkeypatch):
      import <package>.utils.constant as C
      monkeypatch.setattr(C, "DEFAULT_BATCH_SIZE", 2, raising=True)
      ...
  ```

- For integration tests that need environment variations, set env **before** importing the constants module to ensure one-time load semantics:

  ```python
  import importlib, os
  os.environ["DEFAULT_BATCH_SIZE"] = "4"
  import <package>.utils.constant as C
  importlib.reload(C)  # if necessary in the same process
  ```

- Document any new variables in `.env.example` and ensure coverage includes both defaulted and overridden paths.

---

## Final note

If you must deviate (e.g., third-party naming or unavoidable import patterns), add a **short comment** explaining why and keep the ignore as narrow as possible.
