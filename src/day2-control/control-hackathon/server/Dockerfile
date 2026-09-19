FROM python:3.12-slim

WORKDIR /app

# The game logic is the installed `fast` package — the same one the notebooks pull.
ARG FAST_REF=improve-control-hackathon
#main

# git is needed to pip-install the `fast` package from GitHub below.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    "git+https://github.com/sg-ai-safety-hub/FAST.git@${FAST_REF}#subdirectory=src/packages/fast"

COPY app.py engine.py leaderboard.html ./

# The two generated artifacts, both built locally before deploying and shipped in the build context.
# They're gitignored (SecurityEval has no license; house.json holds model-generated insecure
# snippets) but `gcloud run deploy --source .` still uploads them — see .gcloudignore. Building them
# in the image instead would add a build-time network fetch and let house.json drift from a freshly
# re-fetched task suite. Regenerate both before each deploy:
#   python build_tasks.py        # -> tasks.json (curated SecurityEval)
#   python precompute_house.py    # -> house.json (house field, scored against that tasks.json)
# The build fails here if either is missing — the intended guard against deploying a cold server.
COPY tasks.json house.json ./

ENV PORT=8080
# One instance so the in-memory leaderboard is one shared board, not several.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
