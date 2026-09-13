FROM python:3.12-slim

WORKDIR /app

# The game types and constants come from the installed `fast` package — the same one the notebooks
# pull — so the real arena stays a faithful match to the synthetic one from Exercise 1.
ARG FAST_REF=main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    "git+https://github.com/sg-ai-safety-hub/FAST.git@${FAST_REF}#subdirectory=src/packages/fast"

COPY app.py engine.py leaderboard.html tasks.json ./

ENV PORT=8080 ARENA_EPISODES=60
# One instance so the in-memory board is one shared board, not several.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
