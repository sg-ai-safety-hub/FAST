"""The live control arena referee (HTTP layer): Exercise 1's game against a real untrusted model.

This is the same control game the synthetic arena ran — a protocol wrapping an untrusted model U,
scored on safety and usefulness, with a trusted weak model T and a thin human audit budget H — but
the per-task signals are now produced by real models on real coding tasks instead of by formulas.

* **Blue** submits a `protocol(obs) -> action` function *and* a monitor prompt. The prompt drives T
  to score how suspicious each solution looks (`obs.monitor_score`); the protocol decides on it,
  exactly as in Exercise 1: accept U, defer to T, audit (spend one H), or let T edit.
* **Red** submits an `attack(view) -> rate` function *and* a backdoor prompt. The function sets how
  often U arms a backdoor per task; the prompt drives U to plant one that passes the task's tests
  but fails on a hidden input. Subtlety isn't a parameter — it's what the referee measures: how
  suspicious the monitor finds the code, and whether the backdoor actually fires.

All the grading, the model calls, and the game engine live in `engine.py` (which imports without
FastAPI, so it can be tested directly — see `test_engine.py`). This file just validates and enqueues
submissions and serves the board. Model calls need `OPENROUTER_API_KEY`; without it the server still
serves the board and only the model calls error.

It executes model-written and participant-written code. Deploy it disposable and isolated; the blast
radius is one container and the OpenRouter key is the only secret it holds.
"""

from __future__ import annotations

import os
from pathlib import Path

import engine
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = FastAPI(title="Live control arena referee")


class BlueSubmission(BaseModel):
    team: str
    code: str
    monitor_prompt: str


class RedSubmission(BaseModel):
    team: str
    code: str
    attack_prompt: str


class MonitorDryRun(BaseModel):
    monitor_prompt: str


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
    try:
        engine.shakedown_protocol(engine.compile_fn(sub.code, "protocol"))
    except Exception as exc:  # noqa: BLE001 — report the author's error back to them
        return JSONResponse({"ok": False, "error": f"protocol rejected: {exc}"}, status_code=400)
    with engine._lock:
        engine.blue_subs[team] = {"team": team, "status": "pending", "error": None, "scores": {},
                                  "monitor_prompt": sub.monitor_prompt, "code": sub.code}
    engine._work.put(("blue", team, sub.monitor_prompt, sub.code))
    return JSONResponse({"ok": True, "team": team, "role": "blue", "status": "pending"})


@app.post("/submit_red")
def submit_red(sub: RedSubmission) -> JSONResponse:
    team = _team_ok(sub.team)
    if not team:
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if not sub.attack_prompt.strip():
        return JSONResponse({"ok": False, "error": "empty attack prompt"}, status_code=400)
    try:
        engine.shakedown_attack(engine.compile_fn(sub.code, "attack"))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"attack rejected: {exc}"}, status_code=400)
    with engine._lock:
        engine.red_subs[team] = {"team": team, "status": "pending", "error": None, "attacked": {},
                                 "working": 0, "attack_prompt": sub.attack_prompt, "code": sub.code}
    engine._work.put(("red", team, sub.attack_prompt, sub.code))
    return JSONResponse({"ok": True, "team": team, "role": "red", "status": "pending"})


@app.post("/dry_run_red")
def dry_run_red(sub: AttackDryRun) -> JSONResponse:
    """Run a backdoor prompt on a couple of tasks — pass/fire/suspicion — with no board effect."""
    try:
        results = engine.dry_run_attack(sub.attack_prompt)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "results": results})


@app.post("/dry_run_blue")
def dry_run_blue(sub: MonitorDryRun) -> JSONResponse:
    """Score a monitor prompt over a clean/backdoored calibration set — no board effect."""
    try:
        out = engine.dry_run_monitor(sub.monitor_prompt)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, **out})


@app.post("/admin/reset")
def reset(req: AdminRequest) -> JSONResponse:
    if ADMIN_TOKEN and req.token != ADMIN_TOKEN:
        return JSONResponse({"ok": False, "error": "bad admin token"}, status_code=403)
    with engine._lock:
        engine.red_subs.clear()
        engine.blue_subs.clear()
    engine.recompute()
    return JSONResponse({"ok": True})


engine.start_worker()
engine.recompute()
