"""The control-arena referee: stores submissions, cross-scores the whole field, serves the board.

This is the one piece that runs remotely. It keeps every blue protocol and red attack the room
submits, re-runs the full blue-versus-red tournament whenever anything changes, and serves the
leaderboard page that shows the standings. The game logic lives in the installed `fast` package,
so this file is thin: it is an HTTP wrapper plus the arena swap.

This is the Exercise 1 (synthetic) referee: the game runs entirely on `SyntheticArena`, so it makes
no model calls and holds no secrets. The real-model version is a separate service under
`../../live-control/`. Deploy this as a throwaway, isolated container (see README): it executes
participants' submitted policy code to score it — in a restricted namespace, but the container's
isolation is the real boundary.
"""

from __future__ import annotations

import builtins
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from fast.labs.day2_control import control_hackathon as lab

EPISODES = int(os.environ.get("ARENA_EPISODES", "60"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
HERE = Path(__file__).parent

# --- arena ---------------------------------------------------------------------------------------

_state_lock = threading.Lock()
arena = lab.SyntheticArena(seed=0)
submissions: dict[str, dict] = {}  # key "role:team" -> {team, role, code, fn, error}
standings: dict = {"blue": [], "red": [], "fight": None, "updated": 0}


# --- running submitted policies safely ----------------------------------------------------------

_ALLOWED_BUILTINS = [
    "abs", "min", "max", "len", "range", "float", "int", "bool", "round", "sorted", "sum",
    "list", "tuple", "dict", "set", "enumerate", "zip", "map", "filter", "any", "all", "str",
    "print", "True", "False", "None",
]
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}


def compile_policy(code: str, role: str):
    """Turn submitted source into a callable, in a namespace with only the arena's own names.

    No imports, no file or network access — just the action constants and `lab.Attack`. Not a hard
    sandbox; the container's isolation is the real boundary.
    """
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "lab": _lab_surface(),
        "Attack": lab.Attack,
        "ACCEPT": lab.ACCEPT, "DEFER": lab.DEFER, "AUDIT": lab.AUDIT,
        "EDIT": lab.EDIT, "RESAMPLE": lab.RESAMPLE,
    }
    exec(compile(code, "<policy>", "exec"), namespace)  # noqa: S102 — sandboxed, disposable instance
    name = "my_protocol" if role == "blue" else "my_attack"
    fn = namespace.get(name)
    if not callable(fn):
        raise ValueError(f"your code must define a function named {name}(...)")  # noqa: TRY004 — author-facing validation, not a type bug
    return fn


class _LabSurface:
    """The handful of `lab.*` names a policy may touch — not the whole module."""

    Attack = lab.Attack
    ACCEPT, DEFER, AUDIT, EDIT, RESAMPLE = lab.ACCEPT, lab.DEFER, lab.AUDIT, lab.EDIT, lab.RESAMPLE
    MAX_RESAMPLES = lab.MAX_RESAMPLES


def _lab_surface():
    return _LabSurface()


def _safe_safety(blue, red) -> float:
    try:
        return lab.safety(blue, red, arena, episodes=EPISODES)
    except Exception:  # noqa: BLE001 — a broken policy scores as beaten, never crashes the board
        return 0.0


def _safe_usefulness(blue) -> float:
    try:
        return lab.usefulness(blue, arena)
    except Exception:  # noqa: BLE001
        return 0.0


def recompute() -> None:
    """Cross-score the whole field — house teams plus every submission — and publish standings."""
    blue_fns = dict(lab.HOUSE_BLUE)
    red_fns = dict(lab.HOUSE_RED)
    for sub in submissions.values():
        if sub["error"]:
            continue
        (blue_fns if sub["role"] == "blue" else red_fns)[sub["team"]] = sub["fn"]

    house_blue, house_red = set(lab.HOUSE_BLUE), set(lab.HOUSE_RED)
    matrix = {b: {r: _safe_safety(bf, rf) for r, rf in red_fns.items()} for b, bf in blue_fns.items()}

    blue_rows = [
        {
            "team": b, "house": b in house_blue,
            "usefulness": round(_safe_usefulness(bf), 3),
            "worst_case": round(min(matrix[b].values()), 3) if matrix[b] else 1.0,
            "safety": {r: round(v, 3) for r, v in matrix[b].items()},
        }
        for b, bf in blue_fns.items()
    ]
    red_rows = [
        {
            "team": r, "house": r in house_red,
            "worst_for_blue": round(min(matrix[b][r] for b in blue_fns), 3) if blue_fns else 1.0,
            "safety": {b: round(matrix[b][r], 3) for b in blue_fns},
        }
        for r in red_fns
    ]
    blue_rows.sort(key=lambda x: (-x["worst_case"], -x["usefulness"]))
    red_rows.sort(key=lambda x: x["worst_for_blue"])

    fight = None
    if blue_rows and red_rows:
        champ_b, champ_r = blue_rows[0], red_rows[0]
        fight = {
            "blue": champ_b["team"], "red": champ_r["team"],
            "safety": round(matrix[champ_b["team"]][champ_r["team"]], 3),
            "usefulness": champ_b["usefulness"],
        }
    with _state_lock:
        standings.update(blue=blue_rows, red=red_rows, fight=fight, updated=time.time())


# --- HTTP ---------------------------------------------------------------------------------------

app = FastAPI(title="Control Arena referee")


class Submission(BaseModel):
    team: str
    role: str
    code: str


class AdminRequest(BaseModel):
    token: str = ""


@app.get("/", response_class=HTMLResponse)
def board() -> str:
    return (HERE / "leaderboard.html").read_text()


@app.get("/state")
def state() -> JSONResponse:
    with _state_lock:
        return JSONResponse(dict(standings))


@app.post("/submit")
def submit(sub: Submission) -> JSONResponse:
    team = sub.team.strip()[:60]
    if not team or team == "your team name":
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if sub.role not in ("blue", "red"):
        return JSONResponse({"ok": False, "error": "role must be 'blue' or 'red'"}, status_code=400)
    try:
        fn = compile_policy(sub.code, sub.role)
        # a quick shakedown against one house opponent so a broken policy fails loudly here
        if sub.role == "blue":
            lab.safety(fn, lab.HOUSE_RED["blatant"], arena, episodes=4)
        else:
            lab.safety(lab.HOUSE_BLUE["trusted-monitoring"], fn, arena, episodes=4)
    except Exception as exc:  # noqa: BLE001 — report the author's error back to them
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    submissions[f"{sub.role}:{team}"] = {"team": team, "role": sub.role, "code": sub.code, "fn": fn, "error": None}
    recompute()
    return JSONResponse({"ok": True, "team": team, "role": sub.role})


@app.post("/admin/reset")
def reset(req: AdminRequest) -> JSONResponse:
    """Clear the board for a fresh round. The synthetic arena itself is unchanged."""
    if ADMIN_TOKEN and req.token != ADMIN_TOKEN:
        return JSONResponse({"ok": False, "error": "bad admin token"}, status_code=403)
    submissions.clear()
    recompute()
    return JSONResponse({"ok": True})


recompute()  # publish the house-only board at startup
