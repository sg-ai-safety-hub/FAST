#!/usr/bin/env python3
"""Execute solution notebooks and fail if any cell errors.

This is what makes the checks trustworthy. A check that has never been run against a
correct implementation is a coin flip — it might be asserting the wrong thing, or calling
a fixture that no longer exists. Executing every `solution.ipynb` end to end proves the
lab, its fixtures, and its checks all agree.

    uv run python tools/run_notebooks.py              # every solution notebook
    uv run python tools/run_notebooks.py examples/    # just one subtree

The `pip install` cell is skipped: CI installs the working tree with `uv pip install -e .`,
and re-installing from GitHub would test `main` instead of the branch under review.

GPU labs will not run here — GitHub's runners have no GPU. Give those a CPU-scaled path
(see `fast.colab.ci_mode`) or exclude them and run them by hand on Colab before delivery.
Silently skipping a lab is worse than a red build; say which ones are excluded.
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


def main() -> int:
    roots = [ROOT / a for a in sys.argv[1:]] or [ROOT]
    # Everything except lab.ipynb, which is deliberately incomplete and must fail.
    notebooks = sorted(
        {p for r in roots for p in r.rglob("*.ipynb") if p.name != "lab.ipynb"}
    )
    if not notebooks:
        print("no notebooks found — nothing to verify")
        return 0

    os.environ["FAST_CI"] = "1"
    failed = []
    for path in notebooks:
        rel = path.relative_to(ROOT)
        error = run(path)
        if error:
            failed.append((rel, error))
            print(f"FAIL  {rel}\n      {error}")
        else:
            print(f"ok    {rel}")

    print(f"\n{len(notebooks) - len(failed)}/{len(notebooks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
