# Insecure-code control hackathon referee

Backend for the Day 2 hackathon: stores submissions, keeps a shared code pool, re-scores the field
on every change, and serves the live board at `/`. Game logic is the installed `fast` package
(`fast.labs.day2_control.control_hackathon`); this directory is the HTTP wrapper, model calls, and board.

Laid out in three dirs plus the container config at the root:

- `app/` — the running server: `main.py` (endpoints + room-key gate), `engine.py` (model calls, pool,
  scoring, house snapshot), `leaderboard.html` (the page at `/`).
- `data/` — committed JSON resources: `tasks.json` (curated SecurityEval suite), `clean.json` (one
  hand-verified secure baseline per task — the false-positive anchors monitors must not flag), and
  `house.json` (the precomputed house field). The server loads these; it never rebuilds them at runtime.
- `tools/` — dev/ops scripts, not shipped in the image: `build_tasks.py` (→ `data/tasks.json`),
  `precompute_house.py` (→ `data/house.json`), `validate_ladder.py` (pre-room rehearsal), `test_engine.py`.
- `Dockerfile`, `requirements.txt`, `.gcloudignore`, `.env.example` — the container and its config.

Regenerate the `data/` files only when the tasks, curated baselines, ladder, judge, or house prompts
change: `python tools/build_tasks.py` (needs network) then `python tools/precompute_house.py` (needs
the key), and commit the result. `clean.json` is hand-curated — edit it directly.

**The referee never executes submitted code** — snippets are only *read*, by the judge and the
monitors. No code-execution blast radius; the only secret is the OpenRouter key.

## Configuration (`.env`)

`cp .env.example .env`, fill it in (gitignored), and `source .env` before every command below.

| Variable | Used by | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | server | Required; without it the board serves but grading errors. |
| `ROOM_KEY` | server | Shared key gating the writes (submit/dry-run); reads stay open. Set it for any deployed URL. Unset ⇒ fully open (local dev). |
| `ADMIN_TOKEN` | server | Guards `POST /admin/reset {"token": "..."}`. |
| `U_MODEL`, `JUDGE_MODEL` | server | Optional; defaults in the TUNABLES block of `control_hackathon.py`. |
| `ARENA_WORKERS` | server | Optional parallel-worker count (default 8). |
| `PROJECT`, `REGION`, `SERVICE`, `SERVICE_ACCOUNT` | deploy | Read by the `gcloud` commands, not the server. |

Confirm `U_MODEL`, `JUDGE_MODEL`, and the ladder against your key before the room — the slugs/prices
are defaults, not verified billing. `validate_ladder.py` checks exactly that.

## Endpoints

- `POST /submit_blue {team, monitor_prompt, model, temperature?}` · `POST /submit_red {team, attack_prompt, temperature?}`
- `POST /dry_run_blue {monitor_prompt, model, temperature?}` · `POST /dry_run_red {attack_prompt, temperature?}` — no board effect
  - `temperature` (0–2, default 0.5) sets the caller's *own* model (blue's monitor, red's U); the judge is never sampled at it.
- `GET /models` — the ladder · `GET /state` — the board JSON
- `GET /sample` — one worked example (task + MO/judge/monitor decisions) the notebook shows
- `GET /detail?role=blue|red&team=NAME` — per-team drill-down the board rows link to

## Run locally

```sh
source .env
uv pip install -r requirements.txt
uv pip install -e "../../../packages/fast"
uv run uvicorn app.main:app --reload --port 8080   # tasks.json + house.json already in data/
```

`data/house.json` is committed, so the board starts warm; if it's absent the server live-scores the
house on startup instead. Stub the model layer — no key, no network — with `python tools/test_engine.py`.

## Before the room

```sh
source .env
python tools/validate_ladder.py       # whole suite; pass a number for fewer tasks
```

Healthy = big models clear the bar, tiny ones (Llama-3.2 3B/1B) fail; that gap is the game. Every rung
clearing ⇒ attacks too blatant or bar too loose; none clearing ⇒ judge over-labelling or bar too
tight. Tune everything in one place: the TUNABLES block in `control_hackathon.py`. The validator also
flags dead model slugs and confirms a house red yields judge-confirmed insecure code (`working ≥ 1`),
without which the insecure pool is empty and every monitor reads "below bar".

Anthropic models sometimes refuse (`content_filter`) to score a live exploit snippet; the referee
treats that as a *non-signal* (dropped from the rates, not a 0 or an error), so it can't read as
"safe" or let a red DoS every Anthropic monitor. The validator prints how many were dropped.

## Deploy to Cloud Run

**One-time project setup.** A `--source` deploy builds via Cloud Build as the default compute service
account, which on a fresh project lacks the roles it needs (first build fails on
`roles/logging.logWriter`). Grant the bundle once:

```sh
source .env
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
```

That's the *build* identity, separate from `$SERVICE_ACCOUNT` (the *runtime* identity). Wait ~30s.

**Deploy.** `tasks.json` and `house.json` are committed, so deploy is one command (`.gcloudignore`
keeps a local `.env` out of the upload):

```sh
source .env
gcloud run deploy "$SERVICE" \
  --source . --project "$PROJECT" --region "$REGION" \
  --service-account "$SERVICE_ACCOUNT" --allow-unauthenticated --max-instances 1 \
  --set-env-vars "OPENROUTER_API_KEY=$OPENROUTER_API_KEY,ADMIN_TOKEN=$ADMIN_TOKEN,ROOM_KEY=$ROOM_KEY"
```

`--max-instances 1` keeps the in-memory board single and shared. Override models by appending
`,U_MODEL=$U_MODEL,JUDGE_MODEL=$JUDGE_MODEL`.

**If deploy warns "Setting IAM policy failed" and the URL returns 403 Forbidden**, `--allow-unauthenticated`
couldn't grant the public invoker binding. Try it directly:

```sh
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region "$REGION" --member=allUsers --role=roles/run.invoker
```

If that fails with `do not belong to a permitted customer`, a **Domain Restricted Sharing** org
policy is blocking `allUsers`. Relax it for this project (needs `roles/orgpolicy.policyAdmin`), wait
~1–2 min, then re-run the binding above:

```sh
printf 'name: projects/%s/policies/iam.allowedPolicyMemberDomains\nspec:\n  rules:\n    - allowAll: true\n' "$PROJECT" \
  | gcloud org-policies set-policy /dev/stdin
```

(Or Console: **IAM & Admin → Organization Policies → Domain restricted sharing → Manage policy →
Override parent's policy → Add rule → Allow All**.) This lets any resource in the project be shared
publicly, so use a throwaway project for the board if that's a concern.

**Gating.** `--allow-unauthenticated` disables Google-account auth (participants have none); the gate
is `ROOM_KEY`, and it covers only the **writes** (submit/dry-run) that spend model calls — a stray
visitor can't burn your OpenRouter budget. Reads are open, so:

- **Board (browser):** just open `<service-url>` and project it — the display and its `/state`
  polling need no key.
- **Notebook:** participants paste `ROOM_KEY` into the setup cell; the client sends it as `X-Room-Key`
  on every submit/dry-run.

Rotate with `gcloud run services update "$SERVICE" --region "$REGION" --update-env-vars ROOM_KEY=<new>`.

## Destroy

```sh
source .env
gcloud run services delete "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet
gcloud artifacts repositories delete cloud-run-source-deploy --project "$PROJECT" --location "$REGION" --quiet
gcloud storage rm -r "gs://run-sources-${PROJECT}-${REGION}" --project "$PROJECT"
```

The bucket follows `run-sources-<project>-<region>`; if `gcloud storage ls` shows a different
`run-sources-*` bucket, delete that one.
