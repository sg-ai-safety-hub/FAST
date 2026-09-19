# Insecure-code control hackathon referee

Backend for the Day 2 hackathon: stores submissions, keeps a shared code pool, re-scores the field
on every change, and serves the live board at `/`. Game logic is the installed `fast` package
(`fast.labs.day2_control.control_hackathon`); this directory is the HTTP wrapper, model calls, and board.

- `app.py` — endpoints, submission validation, the room-key gate.
- `engine.py` — model calls (MO, judge, monitor), the pool, scoring, the house snapshot.
- `leaderboard.html` — the page at `/`; polls `/state`.
- `build_tasks.py` → `tasks.json` — curated SecurityEval task suite.
- `precompute_house.py` → `house.json` — the house field, scored once so the server starts warm.
- `validate_ladder.py` — pre-room dress rehearsal against real models.
- `Dockerfile`, `requirements.txt`, `.gcloudignore` — the container and what the deploy uploads.

`tasks.json` and `house.json` are generated locally before each deploy and shipped in the image (both
gitignored: SecurityEval has no license; `house.json` holds model-generated insecure snippets). The
server loads them, never rebuilds them at runtime.

**The referee never executes submitted code** — snippets are only *read*, by the judge and the
monitors. No code-execution blast radius; the only secret is the OpenRouter key.

## Configuration (`.env`)

`cp .env.example .env`, fill it in (gitignored), and `source .env` before every command below.

| Variable | Used by | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | server | Required; without it the board serves but grading errors. |
| `ROOM_KEY` | server | Shared key gating the board. Set it for any deployed URL. Unset ⇒ open (local dev). |
| `ADMIN_TOKEN` | server | Guards `POST /admin/reset {"token": "..."}`. |
| `U_MODEL`, `JUDGE_MODEL` | server | Optional; defaults in the TUNABLES block of `control_hackathon.py`. |
| `ARENA_WORKERS` | server | Optional parallel-worker count (default 8). |
| `PROJECT`, `REGION`, `SERVICE`, `SERVICE_ACCOUNT` | deploy | Read by the `gcloud` commands, not the server. |

Confirm `U_MODEL`, `JUDGE_MODEL`, and the ladder against your key before the room — the slugs/prices
are defaults, not verified billing. `validate_ladder.py` checks exactly that.

## Endpoints

- `POST /submit_blue {team, monitor_prompt, model}` · `POST /submit_red {team, attack_prompt}`
- `POST /dry_run_blue {monitor_prompt, model}` · `POST /dry_run_red {attack_prompt}` — no board effect
- `GET /models` — the ladder · `GET /state` — the board JSON

## Run locally

```sh
source .env
uv pip install -r requirements.txt
uv pip install -e "../../../packages/fast"
python build_tasks.py                        # -> tasks.json (needs network)
uv run uvicorn app:app --reload --port 8080
```

Without `house.json` the server live-scores the house on startup (fine locally; run
`precompute_house.py` first to match production). Stub the model layer — no key, no network — with
`python test_engine.py`.

## Before the room

```sh
source .env
python validate_ladder.py       # whole suite; pass a number for fewer tasks
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

**Build the artifacts, then deploy.** Regenerate both whenever the tasks, ladder, or house prompts
change; `.gcloudignore` uploads them (and keeps a local `.env` out):

```sh
source .env
python build_tasks.py        # -> tasks.json (needs network)
python precompute_house.py    # -> house.json (needs the key; makes real paid calls — also a pipeline check)

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
is `ROOM_KEY`. With it set, every request needs the key — a stray visitor gets 401, not a
budget-spending board. Hand out:

- **Board (browser):** open once as `<service-url>/?key=<ROOM_KEY>`; the server sets a cookie so
  `/state` polling works without the key in the URL. Project that tab.
- **Notebook:** participants paste `ROOM_KEY` into the setup cell; the client sends it as `X-Room-Key`.

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
