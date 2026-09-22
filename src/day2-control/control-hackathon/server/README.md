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
| `ARENA_WORKERS` | server | Optional fan-out width for one submission's model calls (default 24). |
| `STATE_PATH` | server | Durable board file (a mounted GCS bucket in prod) so a redeploy restores the room instead of resetting it; set by the deploy command. Unset ⇒ in-memory only. |
| `PROJECT`, `REGION`, `SERVICE`, `SERVICE_ACCOUNT` | deploy | Read by the `gcloud` commands, not the server. |

Confirm `U_MODEL`, `JUDGE_MODEL`, and the ladder against your key before the room — the slugs/prices
are defaults, not verified billing. `validate_ladder.py` checks exactly that.

## Endpoints

- `POST /submit_blue {team, monitor_prompt, model, temperature?}` · `POST /submit_red {team, attack_prompt, temperature?}`
  - blue is keyed by (team, model): a second model from one team adds a row, the same model updates it; red is one row per team
  - one in-flight submission per key: resubmitting a row (same team+model for blue, same team for red) while it's still scoring returns **429**; race several models at once with distinct keys
- `POST /dry_run_blue {monitor_prompt, model, temperature?}` · `POST /dry_run_red {attack_prompt, temperature?}` — no board effect
  - `temperature` (0–2, default 0.5) sets the caller's *own* model (blue's monitor, red's U); the judge is never sampled at it.
- `GET /models` — the ladder · `GET /state` — the board JSON
- `GET /sample` — one worked example (task + MO/judge/monitor decisions) the notebook shows
- `GET /detail?role=blue|red&team=NAME&model=SLUG` — per-row drill-down the board links to (blue rows
  are per (team, model), so `model` targets the right monitor; red is one-per-team and ignores it)

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

**One-time setup.** Grant Cloud Build (the `--source` build identity) its role, create the durable-board
bucket, and let the runtime account read/write it:

```sh
source .env
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
gcloud storage buckets create "gs://$PROJECT-control-board" --project "$PROJECT" --location "$REGION"
gcloud storage buckets add-iam-policy-binding "gs://$PROJECT-control-board" \
  --member="serviceAccount:$SERVICE_ACCOUNT" --role=roles/storage.objectAdmin
```

**Deploy.** One command (`.gcloudignore` keeps a local `.env` out of the upload). The bucket mounts at
`STATE_PATH`'s parent dir and the board lives at `STATE_PATH` inside it (both from `.env`), so a redeploy
restores the live room — every submission — instead of resetting it. To deploy without persistence, drop
the two `--add-volume*` lines and `STATE_PATH` from `--set-env-vars`:

```sh
source .env
gcloud run deploy "$SERVICE" \
  --source . --project "$PROJECT" --region "$REGION" \
  --service-account "$SERVICE_ACCOUNT" --allow-unauthenticated --max-instances 1 \
  --cpu 2 --memory 1Gi --no-cpu-throttling --concurrency 80 \
  --add-volume "name=board,type=cloud-storage,bucket=$PROJECT-control-board" \
  --add-volume-mount "volume=board,mount-path=$(dirname "$STATE_PATH")" \
  --set-env-vars "STATE_PATH=$STATE_PATH,OPENROUTER_API_KEY=$OPENROUTER_API_KEY,ADMIN_TOKEN=$ADMIN_TOKEN,ROOM_KEY=$ROOM_KEY"
```

`--no-cpu-throttling` is load-bearing: scoring runs on a background thread *after* the HTTP response, so
without it the worker stalls between requests. `--max-instances 1` keeps the in-memory board single —
don't raise it. `--cpu` takes whole numbers only (fractional needs throttling). Override models by
appending `,U_MODEL=$U_MODEL,JUDGE_MODEL=$JUDGE_MODEL`. A redeploy on an empty bucket falls back to the
shipped `house.json` (a fresh room).

**Reset the board** — wipes to the fresh house field and clears the saved file (the deliberate way to
start a new room):

```sh
curl -X POST "<service-url>/admin/reset" -H 'Content-Type: application/json' -d "{\"token\":\"$ADMIN_TOKEN\"}"
```

**Throughput.** One worker drains the queue serially and scoring is the monitor × snippet cross-product,
so at ~50 submissions a side the late ones cost minutes and rows sit "pending" (reads stay live).
`ARENA_WORKERS`=24 widens each submission's fan-out, and one-in-flight-per-key returns **429** on a
duplicate resubmit; the next lever, if a room needs it, is capping each red team's pool contribution.

**Public access.** `--allow-unauthenticated` disables Google-account auth (participants have none); the
`ROOM_KEY` gate covers only the writes that spend model calls, so a visitor can't burn the budget. The
board and its `/state` polling are open — open `<service-url>` to project it; participants paste `ROOM_KEY`
into the notebook (sent as `X-Room-Key`). Rotate: `gcloud run services update "$SERVICE" --region "$REGION" --update-env-vars ROOM_KEY=<new>`.

**If the URL returns 403** after a "Setting IAM policy failed" warning, `--allow-unauthenticated` couldn't
grant the public invoker:

```sh
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region "$REGION" --member=allUsers --role=roles/run.invoker
```

If that fails with `do not belong to a permitted customer`, a **Domain Restricted Sharing** org policy
is blocking `allUsers` — relax it for the project (needs `roles/orgpolicy.policyAdmin`), wait ~1–2 min,
re-run the binding:

```sh
printf 'name: projects/%s/policies/iam.allowedPolicyMemberDomains\nspec:\n  rules:\n    - allowAll: true\n' "$PROJECT" \
  | gcloud org-policies set-policy /dev/stdin
```

This shares any resource in the project publicly, so use a throwaway project if that's a concern.

## Custom domain (a stable URL for the notebook)

Map a **subdomain** (DNS can't CNAME an apex) so `SERVER_URL` stays fixed across redeploys. Serve
the parent domain from a Cloud DNS zone, verify it, then map the subdomain — mapping refuses an
unverified domain, and verification is a TXT record you add in the zone.

```sh
source .env

# Make sure dns service is enabled
gcloud services enable dns.googleapis.com --project "$PROJECT"

# In case you need a domain delegation, otherwise pass
gcloud dns managed-zones create securefast-labs --dns-name $DOMAIN_DELEGATION. --visibility public --description control-lab
gcloud dns managed-zones describe securefast-labs --format='value(nameServers)'   # give these to whoever owns securefast.ai to delegate $DOMAIN_DELEGATION, then wait for propagation
gcloud domains verify $DOMAIN_DELEGATION                                                 # opens Search Console; copy the google-site-verification token it shows
gcloud dns record-sets create $DOMAIN_DELEGATION. --zone securefast-labs --type TXT --ttl 300 --rrdatas '"google-site-verification=PASTE_TOKEN"'
# ...re-run `gcloud domains verify $DOMAIN_DELEGATION` once that TXT resolves (dig +short TXT $DOMAIN_DELEGATION), then:

# Verify domain ownership and wire up with the google compute service
gcloud domains verify $DOMAIN
gcloud beta run domain-mappings create --service "$SERVICE" --region "$REGION" --domain "$DOMAIN"
gcloud dns record-sets create "$DOMAIN." --zone securefast-labs --type CNAME --ttl 300 --rrdatas ghs.googlehosted.com.
```

Verifying `$DOMAIN_DELEGATION` covers `control.` under it. Google auto-provisions the TLS cert once DNS
resolves (minutes, occasionally up to ~24h); the `*.run.app` URL keeps working alongside it. An apex
needs 4 A + 4 AAAA records instead, or an external HTTPS load balancer (also needed if domain
mappings aren't offered in `$REGION`).

## Destroy

```sh
source .env
gcloud run services delete "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet
gcloud storage rm -r "gs://$PROJECT-control-board" --project "$PROJECT"                # durable board
gcloud artifacts repositories delete cloud-run-source-deploy --project "$PROJECT" --location "$REGION" --quiet
gcloud storage rm -r "gs://run-sources-$PROJECT-$REGION" --project "$PROJECT"        # build uploads
```

The build-uploads bucket follows `run-sources-<project>-<region>`; if `gcloud storage ls` shows a
different `run-sources-*` bucket, delete that one.
