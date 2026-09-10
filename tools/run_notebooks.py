#!/usr/bin/env python3
"""Execute the notebooks and hold each to what it's meant to do.

Two guarantees, both on every push:

* Every `solution.ipynb` runs end to end without error. Its checks call the reference
  implementation, so a check that disagrees with its own answer fails here rather than in the
  room. This also covers the Day 0 smoke test and the worked template.

* Every `lab.ipynb` fails, and fails *at a check*. The participant notebook ships with the
  exercise bodies removed, so a check that still passes on it is testing nothing (CLAUDE.md).
  Running it and demanding a `CheckFailed` keeps the graders honest: it catches a check so
  loose it would green-light an empty or scaffold answer.

    uv run python tools/run_notebooks.py                 # everything
    uv run python tools/run_notebooks.py src/day1-models  # one subtree

The `pip install` cell is skipped: CI installs the working tree, and reinstalling from GitHub
would test `main` instead of the branch under review.

GPU labs won't run here. GitHub's runners have no GPU, so give those a CPU-scaled path (see
`fast.colab.ci_mode`) or exclude them and run them by hand on Colab before delivery. Say which
ones are excluded; a silently skipped lab is worse than a red build.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 600


def _rel(path: Path) -> Path:
    """Repo-relative path for display, or the path itself if it sits outside the repo."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def executable_cells(nb):
    """Drop install cells — the package is already installed from the working tree."""
    return [c for c in nb.cells if not (c.cell_type == "code" and "pip install" in c.source)]


def run(path: Path) -> str | None:
    """Execute one notebook. Returns an error string, or None on success."""
    nb = nbformat.read(path, as_version=4)
    nb.cells = executable_cells(nb)
    client = NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3", allow_errors=False)
    try:
        client.execute(cwd=str(path.parent))
    except CellExecutionError as exc:
        # The last line alone is often just "AttributeError:" with the detail above it.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        lines = [ln for ln in plain.splitlines() if ln.strip()]
        return " / ".join(lines[-3:])
    except Exception as exc:  # noqa: BLE001 — kernel died, timeout, missing dependency
        return f"{type(exc).__name__}: {exc}"
    return None


def check_solutions(paths: list[Path], failed: list[tuple[Path, str]]) -> None:
    for path in paths:
        rel = _rel(path)
        error = run(path)
        if error:
            failed.append((rel, error))
            print(f"FAIL  {rel}\n      {error}")
        else:
            print(f"ok    {rel}")


def check_stubs(paths: list[Path], failed: list[tuple[Path, str]]) -> None:
    """Each `lab.ipynb` must be rejected by one of its own checks, not by an unrelated crash."""
    for path in paths:
        rel = _rel(path)
        error = run(path)
        if error is None:
            failed.append((rel, "ran clean; a stub notebook must fail its checks"))
            print(
                f"FAIL  {rel}\n      ran to the end without error. Its checks pass on the "
                "removed exercise bodies, so they test nothing"
            )
        elif "CheckFailed" not in error:
            failed.append((rel, f"failed away from a check: {error}"))
            print(
                f"FAIL  {rel}\n      failed, but not at a check (a bug in the provided "
                f"code?): {error}"
            )
        else:
            print(f"ok    {rel}  (rejected by a check, as intended)")


def main() -> int:
    roots = [ROOT / a for a in sys.argv[1:]] or [ROOT]
    notebooks = sorted({p for r in roots for p in r.rglob("*.ipynb")})
    if not notebooks:
        print("no notebooks found — nothing to verify")
        return 0

    solutions = [p for p in notebooks if p.name != "lab.ipynb"]
    stubs = [p for p in notebooks if p.name == "lab.ipynb"]

    os.environ["FAST_CI"] = "1"
    failed: list[tuple[Path, str]] = []

    print("solutions must pass:")
    check_solutions(solutions, failed)
    if stubs:
        print("\nstub notebooks must fail on their own checks:")
        check_stubs(stubs, failed)

    print(f"\n{len(solutions)} solution(s), {len(stubs)} stub(s), {len(failed)} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
