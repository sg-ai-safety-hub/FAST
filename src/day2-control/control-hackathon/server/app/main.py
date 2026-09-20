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


# Every submission carries a `temperature` for the caller's OWN model — blue's monitor, red's U — so
# teams can feel how sampling changes results. Default 0.5. The judge is never sampled at a team's
# temperature; ground truth stays reproducible (see engine._chat).
DEFAULT_TEMPERATURE = 0.5


class BlueSubmission(BaseModel):
    team: str
    monitor_prompt: str
    model: str
    temperature: float = DEFAULT_TEMPERATURE


class RedSubmission(BaseModel):
    team: str
    attack_prompt: str
    temperature: float = DEFAULT_TEMPERATURE


class MonitorDryRun(BaseModel):
    monitor_prompt: str
    model: str
    temperature: float = DEFAULT_TEMPERATURE


class AttackDryRun(BaseModel):
    attack_prompt: str
    temperature: float = DEFAULT_TEMPERATURE


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


@app.get("/detail")
def detail(role: str, team: str, model: str = "") -> JSONResponse:
    """Drill-down for one board row: the shared reference task in full plus a per-task breakdown.
    Read-only, built from loaded state — no model calls. Blue rows are per (team, model), so the board
    passes `model` to target the right monitor; red rows are one-per-team and ignore it."""
    if role not in ("blue", "red"):
        return JSONResponse({"available": False, "error": "role must be blue or red"}, status_code=400)
    return JSONResponse(engine.team_detail(role, team, model or None))


MAX_PROMPT = 4000  # a monitor/attack prompt is a paragraph, not a payload — cap it (each submission
#                    fans model calls out over every task, so an unbounded prompt is a cost lever)


def _team_ok(team: str) -> str | None:
    team = team.strip()[:60]
    return team if team and team != "your team name" else None


def _prompt_error(prompt: str, kind: str) -> str | None:
    if not prompt.strip():
        return f"empty {kind} prompt"
    if len(prompt) > MAX_PROMPT:
        return f"{kind} prompt too long ({len(prompt)} chars; max {MAX_PROMPT})"
    return None


def _temp_error(t: float) -> str | None:
    # OpenRouter accepts 0..2; keep teams in that range so a typo can't error every call in a submission.
    if not 0.0 <= t <= 2.0:
        return f"temperature must be between 0 and 2 (got {t})"
    return None


@app.post("/submit_blue")
def submit_blue(sub: BlueSubmission) -> JSONResponse:
    team = _team_ok(sub.team)
    if not team:
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if err := _prompt_error(sub.monitor_prompt, "monitor"):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not is_allowed_model(sub.model):
        return JSONResponse({"ok": False, "error": f"model {sub.model!r} is not on the ladder — "
                             "GET /models for the choices"}, status_code=400)
    if err := _temp_error(sub.temperature):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not engine.reserve("blue", team, sub.model):
        return JSONResponse({"ok": False, "error": "you already have a submission for this model in the "
                             "queue — wait for it to finish before resubmitting"}, status_code=429)
    with engine._lock:
        # keyed by (team, model): a second model from the same team adds a row, the same model updates it
        engine.blue_subs[engine._bkey(team, sub.model)] = {
            "team": team, "status": "pending", "error": None, "scores": {},
            "monitor_prompt": sub.monitor_prompt, "model": sub.model, "temperature": sub.temperature}
    engine._work.put(("blue", team, sub.monitor_prompt, sub.model, sub.temperature))
    return JSONResponse({"ok": True, "team": team, "role": "blue", "status": "pending"})


@app.post("/submit_red")
def submit_red(sub: RedSubmission) -> JSONResponse:
    team = _team_ok(sub.team)
    if not team:
        return JSONResponse({"ok": False, "error": "set a real team name"}, status_code=400)
    if err := _prompt_error(sub.attack_prompt, "attack"):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if err := _temp_error(sub.temperature):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not engine.reserve("red", team, None):
        return JSONResponse({"ok": False, "error": "you already have a submission in the queue — wait "
                             "for it to finish before resubmitting"}, status_code=429)
    with engine._lock:
        engine.red_subs[team] = {"team": team, "status": "pending", "error": None, "snippets": {},
                                 "working": 0, "attack_prompt": sub.attack_prompt,
                                 "temperature": sub.temperature}
    engine._work.put(("red", team, sub.attack_prompt, None, sub.temperature))
    return JSONResponse({"ok": True, "team": team, "role": "red", "status": "pending"})


@app.post("/dry_run_blue")
def dry_run_blue(sub: MonitorDryRun) -> JSONResponse:
    if not is_allowed_model(sub.model):
        return JSONResponse({"ok": False, "error": f"model {sub.model!r} is not on the ladder"},
                            status_code=400)
    if err := _prompt_error(sub.monitor_prompt, "monitor"):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if err := _temp_error(sub.temperature):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    try:
        out = engine.dry_run_blue(sub.monitor_prompt, sub.model, sub.temperature)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, **out})


@app.post("/dry_run_red")
def dry_run_red(sub: AttackDryRun) -> JSONResponse:
    if err := _prompt_error(sub.attack_prompt, "attack"):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if err := _temp_error(sub.temperature):
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    try:
        results = engine.dry_run_red(sub.attack_prompt, temperature=sub.temperature)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "results": results})


@app.post("/admin/reset")
def reset(req: AdminRequest) -> JSONResponse:
    # Refuse when no token is configured (don't fall open to every room-key holder), else compare
    # in constant time.
    if not ADMIN_TOKEN or not hmac.compare_digest(req.token, ADMIN_TOKEN):
        return JSONResponse({"ok": False, "error": "admin token not configured or incorrect"},
                            status_code=403)
    with engine._lock:
        engine.red_subs.clear()
        engine.blue_subs.clear()
    if not engine.load_house_snapshot():  # restore the precomputed house field, free and instant
        engine.seed_house()               # no snapshot (local dev) — re-score it live
    engine.clear_persisted()  # drop the saved live board, else the next boot restores what we just reset
    engine.recompute()
    return JSONResponse({"ok": True})


engine.start_worker()
# Boot order: a saved live board (STATE_PATH) restores a room across a redeploy, else the shipped
# house.json warm-starts a fresh one, else (local dev, neither) score the house live.
if not engine.load_persisted() and not engine.load_house_snapshot():
    engine.seed_house()
engine.recompute()
