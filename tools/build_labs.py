#!/usr/bin/env python3
"""Build participant and solution notebooks from a single source file.

Each lab is authored once, as a jupytext percent-format `.py` file. This script generates
`lab.ipynb` (what participants open) and `solution.ipynb` (the answer) from it, so the two
can never drift.

    uv run python tools/build_labs.py           # build everything
    uv run python tools/build_labs.py --check   # fail if anything is stale (CI)

Functions participants implement are marked with `@fast.testing.exercise`. The participant
notebook keeps the signature, decorator and docstring, and gets `raise NotImplementedError`
for a body:

    @exercise
    def difference_in_means(harmful, harmless):
        \"\"\"Return the unit-norm direction separating the two sets.\"\"\"
        direction = harmful.mean(axis=0) - harmless.mean(axis=0)   # <- replaced
        return direction / np.linalg.norm(direction)               # <- replaced

Bodies are found by parsing the source, not by scanning for comment markers, so indentation,
nesting and decorators are handled by the parser rather than by us.

Whole cells are marked with a comment on their first line, since a cell is not a Python
construct and has nothing to decorate:

    # @solution-only   → dropped from lab.ipynb
    # @lab-only        → dropped from solution.ipynb
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
import textwrap
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SOURCE_GLOBS = ("day*/**/*.py", "examples/**/*.py")

DECORATOR = "exercise"
SOLUTION_ONLY, LAB_ONLY = "# @solution-only", "# @lab-only"

BANNER = (
    "<!-- generated from {source} — do not edit this notebook directly, "
    "your changes will be overwritten by tools/build_labs.py -->"
)


def _exercise_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """The @exercise decorator node and its stub text, or (None, None) if not marked."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == DECORATOR:
            return decorator, ""
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == DECORATOR
        ):
            for keyword in decorator.keywords:
                if keyword.arg == "stub" and isinstance(keyword.value, ast.Constant):
                    return decorator, textwrap.dedent(str(keyword.value.value)).strip("\n")
            return decorator, ""
    return None, None


def strip_exercise_bodies(source: str) -> str:
    """Replace the body of every @exercise function, keeping signature and docstring."""
    lines = source.splitlines()
    edits: list[tuple[int, int, list[str]]] = []

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        decorator, stub = _exercise_decorator(node)
        if decorator is None:
            continue

        # Normalise `@exercise(stub=...)` to bare `@exercise` — otherwise the participant
        # reads the scaffold twice, once in the decorator and once as the body.
        if stub:
            edits.append(
                (decorator.lineno - 1, decorator.end_lineno, [f"@{DECORATOR}"])
            )

        body = node.body
        start_stmt = 0
        first = body[0]
        is_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_docstring:
            start_stmt = 1  # keep the docstring — it's the spec
        if start_stmt >= len(body):
            continue

        indent = " " * body[start_stmt].col_offset
        replacement = (
            [indent + line for line in stub.splitlines()]
            if stub
            else [indent + "raise NotImplementedError"]
        )
        edits.append((body[start_stmt].lineno - 1, node.end_lineno, replacement))

    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = replacement
    return "\n".join(lines)


def finalise(nb, variant: str, source: str):
    """Drop cells the variant shouldn't have, normalise, and stamp a generated banner."""
    drop = SOLUTION_ONLY if variant == "lab" else LAB_ONLY

    cells, seen = [], set()
    for cell in nb.cells:
        first = cell.source.lstrip().split("\n", 1)[0].strip()
        if first in (SOLUTION_ONLY, LAB_ONLY):
            if first == drop:
                continue
            cell.source = cell.source.split("\n", 1)[1] if "\n" in cell.source else ""

        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)

        # Deterministic, content-derived ids. nbformat's random ids would otherwise put a
        # spurious diff in every rebuild; deriving from content means an edit to one cell
        # doesn't renumber the rest.
        cell_id = hashlib.sha1(cell.source.encode()).hexdigest()[:8]
        while cell_id in seen:
            cell_id = hashlib.sha1(cell_id.encode()).hexdigest()[:8]
        seen.add(cell_id)
        cell.id = cell_id
        cells.append(cell)

    # An HTML comment: invisible in a rendered notebook, plain to anyone who opens the file
    # or reviews a diff. Nothing stops someone editing a generated notebook — this at least
    # tells them the edit won't survive.
    banner = nbformat.v4.new_markdown_cell(BANNER.format(source=source))
    banner.id = "generated-banner"
    nb.cells = [banner, *cells]
    return nb


def outputs_for(source: Path, text: str) -> list[tuple[Path, str]]:
    """Which notebooks a source file produces."""
    has_variants = f"@{DECORATOR}" in text or SOLUTION_ONLY in text or LAB_ONLY in text
    if not has_variants:
        return [(source.with_suffix(".ipynb"), "solution")]
    return [(source.parent / "lab.ipynb", "lab"), (source.parent / "solution.ipynb", "solution")]


def structure(notebook_text: str) -> str:
    """Canonical form of a notebook with outputs and execution counts removed.

    A finished solution is committed *with* its outputs, so it reads on GitHub without a
    GPU. The build always renders outputs stripped, so a byte comparison would call every
    such solution stale. Comparing on structure — cells, source, ids, metadata — lets a
    committed solution keep its outputs while `--check` still guards against the source and
    the notebook drifting apart, which is the only drift that matters here.
    """
    nb = nbformat.reads(notebook_text, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nbformat.writes(nb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if any output is stale")
    args = parser.parse_args()

    sources = sorted(
        p for glob in SOURCE_GLOBS for p in ROOT.glob(glob) if "# %%" in p.read_text()
    )
    if not sources:
        print("no lab sources found")
        return 0

    stale = []
    for source in sources:
        text = source.read_text()
        rel = str(source.relative_to(ROOT))
        for path, variant in outputs_for(source, text):
            variant_text = strip_exercise_bodies(text) if variant == "lab" else text
            nb = finalise(jupytext.reads(variant_text, fmt="py:percent"), variant, rel)
            rendered = jupytext.writes(nb, fmt="ipynb")

            current = path.read_text() if path.exists() else None
            if current is not None and structure(current) == structure(rendered):
                continue  # up to date — and any committed outputs are left untouched
            if args.check:
                stale.append(path.relative_to(ROOT))
            else:
                path.write_text(rendered)
                print(f"  {path.relative_to(ROOT)}")

    if stale:
        print("stale, run `uv run python tools/build_labs.py`:", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"{len(sources)} source file(s) up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
