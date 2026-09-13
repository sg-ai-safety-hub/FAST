# Live control arena referee

The backend for Exercise 2. It plays Exercise 1's control game — protocol vs. attack, scored on
safety and usefulness — but the per-task signals come from real models on real coding tasks. It runs
each red team's backdoor prompt through an untrusted model, grades the generated code in a sandbox
(passes the tests? does the backdoor fire? does a trusted edit strip it?), scores every solution
with each blue team's monitor prompt, and then runs the whole tournament on those cached
measurements. Scoring is a background queue, so submissions show as `scoring…` and resolve.

- `engine.py` — grading, the model layer, and the game engine (a faithful port of
  `control_hackathon`). Imports without FastAPI so it can be tested directly.
- `app.py` — the thin HTTP layer: validates + enqueues submissions, serves the board.
- `test_engine.py` — offline tests (no key, no network): the grader, the sandbox's secret isolation,
  the submit shakedown, and the full scoring chain. CI runs it; run it locally the same way.
- `tasks.json` — the coding-task pool: real APPS interview problems (verified references + tests)
  plus the `difficulty` / `benign` features red's attack conditions on. Chosen so the trusted model
  fails several while the reference (honest U) is correct — that gap is what makes deferring cost
  usefulness. Swap the pool by replacing this file (same schema).
- `leaderboard.html` — the scoreboard at `/`; polls `/state`. Submissions come from the notebook,
  which also has pre-submit dry-runs: `/dry_run_red` (does a backdoor pass/fire, how suspicious) and
  `/dry_run_blue` (score a monitor prompt over a clean/backdoored calibration set).
- `Dockerfile`, `requirements.txt` — the container. Installs `fast` from the public repo for the
  shared game types.

## Deploy to Cloud Run

```sh
gcloud run deploy live-control-arena \
  --source . --region <your-region> --allow-unauthenticated --max-instances 1 \
  --set-env-vars OPENROUTER_API_KEY=<key>,ADMIN_TOKEN=<secret>
```

This referee **needs an OpenRouter key** — it calls a capable untrusted model `U_MODEL` to write
solutions and a smaller trusted model `T_MODEL` to monitor, solve, and edit (both overridable by env
var). Defaults validated against OpenRouter, Sept 2026: `U_MODEL=deepseek/deepseek-chat` (complies
with backdoor-writing and returns parseable code) and `T_MODEL=qwen/qwen-2.5-7b-instruct` (separates
clean from backdoored code, yet is genuinely weaker than U on hard tasks). Cost scales with
submissions × tasks; with a handful of teams and cached scoring it stays modest. Pin
`--max-instances 1` (the board is in-memory). `ADMIN_TOKEN` guards `POST /admin/reset`, which clears
the board for a fresh round.

## Run it locally

```sh
uv pip install -r requirements.txt
OPENROUTER_API_KEY=<key> uv run uvicorn app:app --reload --port 8080   # from this directory
```

Without a key the server still starts and serves the board; only the model calls error. The grading
and game engine can be exercised offline by stubbing the model layer — override `app.generate_backdoor`,
`app.monitor_score`, `app.trusted_solution` and `app.trusted_edit`.

## Safety

The referee **executes model-written code** to grade it (subprocess, with a timeout) and **execs
participants' protocol/attack functions** to score them (in a namespace with no imports and a
handful of builtins — not a hard sandbox against a determined attacker, and this is a room of them).
Deploy it as a disposable, isolated instance with nothing else in it; the blast radius is that one
container, and the OpenRouter key is the only secret it holds.
