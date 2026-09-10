"""Shared lab utilities for FAST — Frontier AI Security Training.

Every lab notebook opens with the same two cells:

    <install cell — pip installs this package from GitHub>
    from fast.colab import setup; setup()

The install cell pip-installs `git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast`. While
the repo is private it authenticates with a `GITHUB_TOKEN` read from the environment (locally)
or Colab Secrets, so the same notebook runs in either place. Pin `@main` to a release tag
before the program runs, so a commit on Wednesday morning can't change the environment under a
lab running Wednesday afternoon.
"""

__version__ = "0.1.0"
