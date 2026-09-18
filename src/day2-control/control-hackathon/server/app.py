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

import os
from pathlib import Path

import engine
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from fast.labs.day2_control.control_hackathon import MODEL_LADDER, is_allowed_model

HERE = Path(__file__).parent
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = FastAPI(title="Insecure-code control hackathon referee")


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
    engine.seed_house()
    engine.recompute()
    return JSONResponse({"ok": True})


engine.start_worker()
engine.seed_house()
engine.recompute()
