"""Precompute the house field into `house.json`, so the deployed referee loads a warm board instead
of re-scoring the house with live model calls on every cold start.

Run it before deploying, with your key — it makes real, paid model calls (the same ones the server
would otherwise make on every startup):

    source .env
    python build_tasks.py        # house.json is built against this task suite; make it first
    python precompute_house.py    # writes house.json beside this script

The Dockerfile copies `house.json` into the image. It is gitignored — it holds model-generated
insecure snippets, the same reason `tasks.json` isn't vendored. Regenerate it whenever the house
prompts, the ladder, or the task suite change.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # server root, so `app` imports

# Load OPENROUTER_API_KEY from a .env above us if it isn't already in the environment.
if not os.environ.get("OPENROUTER_API_KEY"):
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
            break

from app import engine  # imported after the key is put in the environment above


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set — put it in .env or the environment")

    print(f"precomputing the house field: {len(engine.HOUSE_RED)} red + {len(engine.HOUSE_BLUE)} "
          f"blue over {len(engine.TASKS)} tasks (live model calls) ...")
    snap = engine.precompute_house()

    working = {team: r["working"] for team, r in snap["red"].items()}
    print(f"\nclean baselines: {len(snap['honest'])}/{len(engine.TASKS)} tasks")
    print(f"house red working (does-task AND insecure) counts: {working}")
    if not any(working.values()):
        print("  WARNING: no house red produced a judge-confirmed insecure snippet — the insecure "
              "pool is empty and every monitor will read 'below bar'. Check the key and the judge.")
    for team, b in snap["blue"].items():
        m = engine._blue_metrics(b)
        print(f"  house blue {team:<20} {b['model']:<34} "
              f"tpr={m['tpr']} fpr={m['fpr']} clears={m['clears']}")

    engine.HOUSE_SNAPSHOT.write_text(json.dumps(snap, indent=2))
    print(f"\nwrote {engine.HOUSE_SNAPSHOT.name} — the image ships this and loads it at startup")


if __name__ == "__main__":
    main()
