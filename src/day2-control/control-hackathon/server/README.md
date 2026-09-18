# Insecure-code control hackathon referee

The remote backend for the Day 2 hackathon. It stores every team's submission, keeps a shared pool
of code, re-scores the whole field whenever anything changes, and serves the live leaderboard at `/`.
The game logic is the installed `fast` package (`fast.labs.day2_control.control_hackathon`); this
directory is the HTTP wrapper, the model calls, and the board.

- `app.py` — endpoints and submission validation.
- `engine.py` — the model calls (MO, judge, monitor), the pool, and the scoring.
- `leaderboard.html` — the page served at `/`; it polls `/state`.
- `build_tasks.py` — curates the task suite from SecurityEval into `tasks.json` at build time.
- `Dockerfile`, `requirements.txt` — the container.

**The referee never executes submitted code.** Red's snippets are only ever *read* — by the judge
model and by the monitors. There is no code-execution blast radius; the only secret in the room is
the OpenRouter key, and the grading path never exposes it.

## What it needs

- `OPENROUTER_API_KEY` — required for the model calls. Without it the board still serves and only the
  model calls error.
- `U_MODEL`, `JUDGE_MODEL` — optional overrides for the untrusted author and the judge (defaults in
  `control_hackathon.py`). **Confirm these and the blue model ladder against your key** — the slugs
  and prices there are intended defaults, not verified billing.
- `ADMIN_TOKEN` — guards `POST /admin/reset {"token": "..."}`, which clears the board (and re-seeds
  the house teams) for a fresh round.

## Endpoints

- `POST /submit_blue {team, monitor_prompt, model}` · `POST /submit_red {team, attack_prompt}`
- `POST /dry_run_blue {monitor_prompt, model}` · `POST /dry_run_red {attack_prompt}` — no board effect
- `GET /models` — the model ladder blue picks from · `GET /state` — the board JSON

## Deploy to Cloud Run

One command from this directory. The image build runs `build_tasks.py`, so the deployed server has
the task suite without the repo carrying it.

```sh
gcloud run deploy control-hackathon \
  --source . \
  --region <your-region> \
  --allow-unauthenticated \
  --max-instances 1 \
  --set-env-vars OPENROUTER_API_KEY=<key>,ADMIN_TOKEN=<pick-a-secret>
```

`--max-instances 1` matters: the board lives in the instance's memory, so one instance keeps it a
single shared board. Hand the printed URL to the room — that's the leaderboard.

## Run it locally

```sh
uv pip install -r requirements.txt
uv pip install -e "../../../packages/fast"   # the game logic; the server imports it
python build_tasks.py                        # fetch + curate tasks.json (needs network)
OPENROUTER_API_KEY=<key> uv run uvicorn app:app --reload --port 8080   # from this directory
```

Test the scoring engine with the model layer stubbed — no key, no network:

```sh
python test_engine.py
```

## Before the room

Run the validator with your key — it exercises the real pipeline and prints where the clear/fail
boundary lands, so you confirm the lesson holds before anyone's in the seats:

```sh
OPENROUTER_API_KEY=<key> python validate_ladder.py     # whole suite; pass a number for fewer tasks
```

A healthy result has the big model(s) clearing and the tiny ones (Llama-3.2 3B/1B) failing — that
gap is the whole game, and it's the thing to confirm before the room. If every rung clears, the
attacks are too blatant (sharpen the house red prompts) or the bar is too loose; if none clears, the
judge is over-labelling or the bar is too tight. Tune it all in one place: the TUNABLES block in
`fast/labs/day2_control/control_hackathon.py` (ladder, models, thresholds, token budgets). The
validator also surfaces dead model slugs and confirms a house red produces judge-confirmed insecure
code (`working ≥ 1`) — without that, the insecure pool is empty and every monitor reads "below bar".

**A note on content filters.** Anthropic models occasionally refuse ("content_filter") to score a
snippet that is itself a live exploit gadget. The referee treats such a call as a *non-signal* —
that (monitor, snippet) pair is dropped from the rates, not scored 0 and not an error — so one
filter-tripping snippet can't silently read as "safe" or let a red DoS every Anthropic-based
monitor. The validator prints how many were dropped per model.
