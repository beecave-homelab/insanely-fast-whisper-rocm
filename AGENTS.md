# AGENTS.md — Coding Rules (Ruff-based)

This repo uses **Ruff** as the source of truth for linting. If your change violates these rules, CI will fail.
Run locally before committing:

```bash
pdm run ruff check --fix .
pdm run ruff format .
```

When in doubt, prefer **correctness → clarity → consistency → brevity** (in that order).

---

## 1) Correctness (Ruff `F` — Pyflakes)

### What It Enforces (Correctness)

* No **undefined names** or variables.
* No **unused imports/variables/arguments**.
* No **duplicate arguments** in function definitions.
* No `import *`.

### Agent Checklist (Correctness)

* Delete dead code and unused symbols.
* Keep imports minimal and explicit.
* If a variable is only used in a comprehension, don’t keep it outside the scope.

---

## 2) PEP 8 surface rules (Ruff `E`, `W` — pycodestyle)

### What It Enforces (PEP 8 surface rules)

* Basic spacing/blank-line/indentation hygiene.
* No trailing whitespace.
* Reasonable line breaks; respect the project’s configured line length (see `ruff.toml`).

### Agent Checklist (PEP 8 surface rules)

* Let the formatter handle whitespace; don’t fight it.
* Break long expressions cleanly (after operators, around commas, etc.).
* Keep files ending with a single newline.

---

## 3) Naming conventions (Ruff `N` — pep8-naming)

### What It Enforces (Naming conventions)

* `snake_case` for functions, methods, and non-constant variables.
* `CapWords` (PascalCase) for classes.
* `UPPER_CASE` for module-level constants.
* Exception classes named `SomethingError` and subclass `Exception`.

### Agent Checklist (Naming conventions)

* Don’t introduce camelCase unless mirroring a third-party API; if you must, add a local pragma to silence `N` for that line only.

---

## 4) Imports: order & style (Ruff `I` — isort rules)

### What It Enforces (Imports: order & style)

* Imports grouped as:

  1. **Standard library**, 2) **Third-party**, 3) **First-party/local**.
* **Alphabetical** within groups; one blank line **between** groups.
* Prefer **one import per line** for clarity.

### Agent Checklist (Imports: order & style)

* Keep all imports **top-of-file** (module scope).
* Don’t alias unless it **adds clarity** (e.g., `import numpy as np`).

### Canonical example

```python
from __future__ import annotations

import dataclasses
import pathlib

import httpx
import pydantic

from mypkg.core import config
from mypkg.utils.paths import ensure_dir
```

---

## 5) Docstrings — content & style (Ruff `D` + `DOC`)

This codebase **requires docstrings** for public modules, classes, functions, and methods. Ruff enforces both **pydocstyle** (`D…`) and **pydoclint** (`DOC…`) style/consistency checks.

**Single-source style**: **Google-style** docstrings with type hints in signatures.

### Rules of thumb

* Triple double quotes.
* First line: **one-sentence summary**, capitalized, ends with a period.
* Blank line after summary; then details.
* Keep **Args/Returns/Raises** sections **in sync** with the signature (pydoclint will fail if mismatched).
* Use **imperative mood** (“Return…”, “Validate…”) and avoid repetition of obvious types (use the type hints).

### Function template

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

### Class template

```python
class ResourceManager:
    """Coordinate creation and lifecycle of resources.

    Notes:
        Thread-safe for read operations; writes are serialized.
    """
```

---

## 6) Import hygiene (Ruff `TID` — flake8-tidy-imports)

### What It Enforces (Import hygiene)

* Prefer **absolute imports** over deep relative imports.
* Avoid circular imports by organizing modules; don’t import inside functions unless necessary for performance or to break a cycle.
* Avoid re-exporting large surfaces implicitly; if you re-export, do it explicitly via `__all__`.

### Agent Checklist (Import hygiene)

* Use `from pkg.subpkg import thing` (absolute), **not** `from .subpkg import thing`, unless it’s the clear local intent and passes lint.
* Gate optional imports like this:

```python
try:
    import rich
except ModuleNotFoundError:  # pragma: no cover
    rich = None  # type: ignore[assignment]
```

---

## 7) Modern Python upgrades (Ruff `UP` — pyupgrade)

### What It Enforces / Prefers (Modern Python upgrades)

* **f-strings** over `format()` / `%` formatting.
* Built-in *PEP 585* generics (`list[str]`, `dict[str, int]`) over `typing.List`, `typing.Dict`, etc.
* **Context managers** where appropriate.
* Remove legacy constructs (`six`, old `u''` prefixes, redundant `object` inheritance, etc.).

### Agent Checklist (Modern Python upgrades)

* Prefer `pathlib.Path` to raw string paths.
* Prefer assignment expressions (`:=`) **sparingly** when it improves clarity.
* Replace `if x == None` → `if x is None`.

---

## 8) Future annotations (Ruff `FA` — flake8-future-annotations)

### What It Enforces (Future annotations)

* Each module must begin with:

```python
from __future__ import annotations
```

### Agent Checklist (Future annotations)

* Place it at the **very top**, after the encoding line (if any) and before all other imports.
* Don’t add it twice; Ruff will tell you.

---

## 9) Local ignores (only when justified)

Prefer fixing the root cause. If a one-off ignore is truly necessary, keep it **scoped and documented**:

```python
value = compute()  # noqa: F401  # used by plugin loader via reflection
```

For docstrings mismatches caused by third-party constraints, prefer a targeted `# noqa: D, DOC…` on that line or block with a brief reason.

---

## 10) Tests & examples

* Tests must follow the same rules as production code.
* Test names: `test_<unit_under_test>__<expected_behavior>()`.
* Docstring examples should be runnable when practical:

```python
def add(a: int, b: int) -> int:
    """Return the sum of two integers.

    Examples:
        >>> add(2, 3)
        5
    """
```

---

## 11) Commit discipline (quick reminder)

Keep diffs clean by running Ruff before committing. Use the project’s conventional commit format. Small, focused commits make lint errors easier to spot.

---

## 12) Quick DO / DON’T

### DO

* Write Google-style docstrings that match signatures.
* Use absolute imports and sorted import blocks.
* Use f-strings and modern type syntax (`list[str]`).
* Remove unused code promptly.

### DON’T

* Introduce camelCase names (except mirroring external APIs).
* Use `import *` or deep relative imports.
* Leave parameters undocumented in public functions.
* Add broad `noqa` comments—always scope them.

---

## 13) Pre-commit (recommended)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9  # keep in sync with ruff version
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## 14) CI expectations

CI will run the same two commands:

```bash
ruff check .
ruff format --check .
```

A PR is mergeable only when both pass.

---

### Final note

If you need to deviate (e.g., third-party naming or unavoidable import patterns), add a **short comment** explaining why and keep the ignore as narrow as possible.
