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

# app/ = the server (main.py, engine.py, leaderboard.html); data/ = committed tasks.json + house.json.
COPY app ./app
COPY data ./data

ENV PORT=8080
# One instance so the in-memory leaderboard is one shared board, not several.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
