FROM python:3.12-slim

WORKDIR /app

# The game logic is the installed `fast` package — the same one the notebooks pull.
ARG FAST_REF=main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    "git+https://github.com/sg-ai-safety-hub/FAST.git@${FAST_REF}#subdirectory=src/packages/fast"

COPY app.py engine.py leaderboard.html build_tasks.py ./
# Curate the task suite from SecurityEval at build time — it is not vendored in the repo.
RUN python build_tasks.py

ENV PORT=8080
# One instance so the in-memory leaderboard is one shared board, not several.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
