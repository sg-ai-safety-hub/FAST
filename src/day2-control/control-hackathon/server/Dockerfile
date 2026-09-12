FROM python:3.12-slim

WORKDIR /app

# The game logic is the installed `fast` package — the same one the notebooks pull. While the repo
# is private, pass a token: `--build-arg GITHUB_TOKEN=...`; drop it once the repo is public.
ARG GITHUB_TOKEN=
ARG FAST_REF=main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN AUTH=$( [ -n "$GITHUB_TOKEN" ] && echo "${GITHUB_TOKEN}@" || echo "" ); \
    pip install --no-cache-dir \
    "git+https://${AUTH}github.com/sg-ai-safety-hub/FAST.git@${FAST_REF}#subdirectory=src/packages/fast"

COPY app.py model_arena.py leaderboard.html ./

ENV PORT=8080 ARENA_EPISODES=60
# One instance so the in-memory leaderboard is one shared board, not several.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
