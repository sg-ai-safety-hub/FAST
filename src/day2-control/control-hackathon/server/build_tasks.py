"""Curate the arena's task suite from SecurityEval — run at deploy, not committed.

The tasks are Python function stubs drawn from **SecurityEval** (Siddiq & Santos, MSR4P&S 2022,
https://github.com/s2e-lab/SecurityEval): each is a short signature + docstring describing a
security-sensitive operation (run a shell command, open a user-supplied path, build a SQL query,
deserialize bytes, hash a secret). A model can complete any of them securely or insecurely — which
is exactly the choice red's model organism gets to make and blue's monitor has to catch.

SecurityEval ships no licence file, so we **do not vendor its prompts** into this public repo
(see CLAUDE.md: datasets are referenced by source, not committed). Instead this script fetches the
dataset and writes a curated `tasks.json` next to it at deploy time; that file is gitignored. The
Dockerfile runs this during image build, so the deployed server has the tasks without the repo
carrying them.

    python build_tasks.py            # writes tasks.json
    python build_tasks.py --check    # just report what would be selected

We keep only a handful of high-signal CWEs and a few tasks each: a tight, judge-legible target beats
broad coverage for a timeboxed hackathon.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "tasks.json"
SOURCE = "https://raw.githubusercontent.com/s2e-lab/SecurityEval/main/dataset.jsonl"

# The CWEs we play on, in board order, with the plain name the judge and the board show. Chosen so
# the vulnerability is obvious once seen and hard to hide in plain code — good red/blue tension.
CWES = {
    "CWE-078": "OS command injection",
    "CWE-089": "SQL injection",
    "CWE-022": "Path traversal",
    "CWE-502": "Unsafe deserialization",
    "CWE-327": "Weak cryptography",
}
MAX_PER_CWE = 3


def _cwe_of(sample_id: str) -> str:
    """'CWE-078_author_1.py' -> 'CWE-078'."""
    return "-".join(sample_id.split("-")[:2]).split("_")[0]


def fetch() -> list[dict]:
    req = urllib.request.Request(SOURCE, headers={"User-Agent": "fast-build-tasks"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def curate(rows: list[dict]) -> list[dict]:
    by_cwe: dict[str, list[dict]] = {c: [] for c in CWES}
    for row in rows:
        cwe = _cwe_of(row["ID"])
        if cwe in by_cwe and len(by_cwe[cwe]) < MAX_PER_CWE:
            by_cwe[cwe].append(row)
    tasks = []
    for cwe, name in CWES.items():  # board order
        for row in by_cwe[cwe]:
            tasks.append({
                "id": row["ID"].removesuffix(".py"),
                "cwe": cwe,
                "cwe_name": name,
                "prompt": row["Prompt"].rstrip() + "\n",
            })
    return tasks


def main() -> None:
    tasks = curate(fetch())
    if "--check" in sys.argv:
        for t in tasks:
            print(f"  {t['id']:24s} {t['cwe']} {t['cwe_name']}")
        print(f"{len(tasks)} tasks across {len({t['cwe'] for t in tasks})} CWEs")
        return
    OUT.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {OUT}")


if __name__ == "__main__":
    main()
