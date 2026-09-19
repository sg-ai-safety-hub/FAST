"""The insecure-code control hackathon referee (HTTP layer).

Validates and enqueues submissions and serves the board; all the model calls, grading, and scoring
live in `engine.py` (which imports without FastAPI, so it can be tested directly — see
`test_engine.py`).

* **Blue** submits a monitor prompt and a model from the ladder. The prompt drives that model to rate
  each snippet's suspicion; the board reads off caught-rate and false-flag-rate and ranks blue by the
  cheapest model that still clears the bar.
* **Red** submits an attack prompt that turns `U_MODEL` into a model organism writing insecure code;
  a judge model decides whether each result actually does the task and is actually insecure, and the
  board ranks red by how many monitors it fools.

The referee never executes submitted code — it is only read by the judge and the monitors. Model
calls need `OPENROUTER_API_KEY`; without it the board still serves and only the model calls error.
Deploy it disposable and isolated all the same; the OpenRouter key is the only secret it holds.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app import engine
from fast.labs.day2_control.control_hackathon import MODEL_LADDER, is_allowed_model

HERE = Path(__file__).parent
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
# A shared room key gates the *writes* — submit and dry-run, the endpoints that spend model calls —
# so a public URL can't be used to burn the OpenRouter budget. Reads (the board, /state, /models) are
# open: they only display. Unset ⇒ fully open, for local dev. The notebook sends the key as a header.
ROOM_KEY = os.environ.get("ROOM_KEY", "")

app = FastAPI(title="Insecure-code control hackathon referee")


def _presented_key(request: Request) -> str:
    """The key a caller offers on a write. The notebook sends it as an `X-Room-Key` header."""
    header = request.headers.get("x-room-key")
    if header:
        return header
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return ""


@app.middleware("http")
async def require_room_key(request: Request, call_next):
    # Only gate methods that mutate/cost (POST). GET/HEAD/OPTIONS — the board and its polling — stay open.
    if (ROOM_KEY and request.method not in ("GET", "HEAD", "OPTIONS")
            and not hmac.compare_digest(_presented_key(request), ROOM_KEY)):
        return JSONResponse({"ok": False, "error": "room key required"}, status_code=401)
    return await call_next(request)


class BlueSubmission(BaseModel):
    team: str
    monitor_prompt: str
    model: str


class RedSubmission(BaseModel):
    team: str
    attack_prompt: str


class MonitorDryRun(BaseModel):
    monitor_prompt: str
    model: str


class AttackDryRun(BaseModel):
    attack_prompt: str


class AdminRequest(BaseModel):
    token: str = ""


@app.get("/", response_class=HTMLResponse)
def board() -> str:
    return (HERE / "leaderboard.html").read_text()


@app.get("/state")
def state() -> JSONResponse:
    with engine._lock:
        return JSONResponse(dict(engine.standings))


@app.get("/models")
def models() -> JSONResponse:
    """The ladder blue picks from, dearest first — the board and the notebook both read this."""
    return JSONResponse({"ladder": MODEL_LADDER})


@app.get("/sample")
def sample() -> JSONResponse:
    """One worked example (a real task + the MO/judge/monitor decisions on it) so the notebook can
    show participants the pipeline. Read-only, built from the house field — no model calls."""
    return JSONResponse(engine.sample_case())


def _team_ok(team: str) -> str | None:
    team = team.strip()[:60]
    return team if team and team != "your team name" else None


@app.post("/submit_blue")
def submit_blue(sub: BlueSubmission) -> JSONResponse:
    team = _team_ok(sub.team)
    if not team:
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if not sub.monitor_prompt.strip():
        return JSONResponse({"ok": False, "error": "empty monitor prompt"}, status_code=400)
    if not is_allowed_model(sub.model):
        return JSONResponse({"ok": False, "error": f"model {sub.model!r} is not on the ladder — "
                             "GET /models for the choices"}, status_code=400)
    with engine._lock:
        engine.blue_subs[team] = {"team": team, "status": "pending", "error": None, "scores": {},
                                  "monitor_prompt": sub.monitor_prompt, "model": sub.model}
    engine._work.put(("blue", team, sub.monitor_prompt, sub.model))
    return JSONResponse({"ok": True, "team": team, "role": "blue", "status": "pending"})


@app.post("/submit_red")
def submit_red(sub: RedSubmission) -> JSONResponse:
    team = _team_ok(sub.team)
    if not team:
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if not sub.attack_prompt.strip():
        return JSONResponse({"ok": False, "error": "empty attack prompt"}, status_code=400)
    with engine._lock:
        engine.red_subs[team] = {"team": team, "status": "pending", "error": None, "snippets": {},
                                 "working": 0, "attack_prompt": sub.attack_prompt}
    engine._work.put(("red", team, sub.attack_prompt, None))
    return JSONResponse({"ok": True, "team": team, "role": "red", "status": "pending"})


@app.post("/dry_run_blue")
def dry_run_blue(sub: MonitorDryRun) -> JSONResponse:
    if not is_allowed_model(sub.model):
        return JSONResponse({"ok": False, "error": f"model {sub.model!r} is not on the ladder"},
                            status_code=400)
    try:
        out = engine.dry_run_blue(sub.monitor_prompt, sub.model)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, **out})


@app.post("/dry_run_red")
def dry_run_red(sub: AttackDryRun) -> JSONResponse:
    try:
        results = engine.dry_run_red(sub.attack_prompt)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "results": results})


@app.post("/admin/reset")
def reset(req: AdminRequest) -> JSONResponse:
    if ADMIN_TOKEN and req.token != ADMIN_TOKEN:
        return JSONResponse({"ok": False, "error": "bad admin token"}, status_code=403)
    with engine._lock:
        engine.red_subs.clear()
        engine.blue_subs.clear()
    if not engine.load_house_snapshot():  # restore the precomputed house field, free and instant
        engine.seed_house()               # no snapshot (local dev) — re-score it live
    engine.recompute()
    return JSONResponse({"ok": True})


engine.start_worker()
# The house field is precomputed into house.json (see precompute_house.py) and shipped in the image,
# so a cold start loads a warm board instead of re-scoring the house with live model calls every time.
if not engine.load_house_snapshot():
    engine.seed_house()  # fallback for local dev without a snapshot: score the house live
engine.recompute()
