# Control Arena referee

The remote backend for the Day 2 hackathon. It stores every team's submission, re-runs the full
blue-versus-red tournament whenever anything changes, and serves the live leaderboard at `/`. The
game logic is the installed `fast` package; this directory is just the HTTP wrapper.

- `app.py` — endpoints, the cross-scoring loop, the arena swap.
- `model_arena.py` — the real-model arena (a real monitor over OpenRouter).
- `leaderboard.html` — the page served at `/`; it polls `/state` and posts to `/submit`.
- `Dockerfile`, `requirements.txt` — the container.

## Deploy to Cloud Run

One command from this directory. It builds the container and gives you a public URL.

```sh
gcloud run deploy control-arena \
  --source . \
  --region <your-region> \
  --allow-unauthenticated \
  --max-instances 1 \
  --set-env-vars ADMIN_TOKEN=<pick-a-secret>
```

`--max-instances 1` matters: the board lives in the instance's memory, so one instance keeps it a
single shared board. Hand the printed URL to the room — that's the leaderboard. While the repo is
private, the image build needs a token to install the `fast` package; add
`--build-arg GITHUB_TOKEN=<token>` (or make the repo public first and drop it).

`ADMIN_TOKEN` guards one endpoint: `POST /admin/reset {"token": "..."}` clears the board for a fresh
round.

## The arena mode is a deploy-time choice

The board lives in the container's memory, so the arena is fixed when the instance starts
(`ARENA_MODE`, default `synthetic`) and never changes under a running board — re-scoring everyone's
submissions on a different arena mid-session would silently reshuffle standings they can't see. To
move to a real model partway through the afternoon, deploy a fresh instance in model mode and have
teams re-submit (one notebook cell):

```sh
gcloud run deploy control-arena \
  --source . --region <your-region> --allow-unauthenticated --max-instances 1 \
  --set-env-vars ARENA_MODE=model,OPENROUTER_API_KEY=<key>,ADMIN_TOKEN=<secret>
```

A fresh deploy starts with a clean board, which is the point — the synthetic round and the
real-model round are separate competitions. The monitor model is `MONITOR_MODEL` (default a small
open model on OpenRouter); the key is read only while building the task table at startup, never in
the path that runs submitted policy code. Note the board is in-memory: pin `--max-instances 1`, and
if you need it to survive an instance restart, back it with Firestore or a GCS object (not built
in yet).

## Run it locally

```sh
uv pip install -r requirements.txt
uv run uvicorn app:app --reload --port 8080   # from this directory
```

## A note on running submitted code

`/submit` executes each team's policy to score it, in a namespace with no imports and no file or
network access. That is not a hard sandbox against a determined attacker, and this is a room full
of them. Deploy the referee as a disposable, isolated instance with no secrets beyond the
OpenRouter key (which never shares the process that runs policies), and treat the blast radius as
that one container.
