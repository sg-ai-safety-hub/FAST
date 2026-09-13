# Control Arena referee

The remote backend for the Day 2 hackathon. It stores every team's submission, re-runs the full
blue-versus-red tournament whenever anything changes, and serves the live leaderboard at `/`. The
game logic is the installed `fast` package; this directory is just the HTTP wrapper.

- `app.py` — endpoints, the cross-scoring loop.
- `leaderboard.html` — the page served at `/`; it polls `/state` and posts to `/submit`.
- `Dockerfile`, `requirements.txt` — the container.

This is the backend for Exercise 1, the synthetic policy game. Exercise 2 (real models in the loop)
has its own backend under `../live-control/`.

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
single shared board. Hand the printed URL to the room — that's the leaderboard. The image build
installs the `fast` package straight from the public repo.

`ADMIN_TOKEN` guards one endpoint: `POST /admin/reset {"token": "..."}` clears the board for a fresh
round. The board is in-memory, so pin `--max-instances 1`; if you need it to survive an instance
restart, back it with Firestore or a GCS object (not built in yet).

## Run it locally

```sh
uv pip install -r requirements.txt
uv run uvicorn app:app --reload --port 8080   # from this directory
```

## A note on running submitted code

`/submit` executes each team's policy to score it, in a namespace with no imports and no file or
network access. That is not a hard sandbox against a determined attacker, and this is a room full
of them. Deploy the referee as a disposable, isolated instance with no secrets in it, and treat the
blast radius as that one container.
