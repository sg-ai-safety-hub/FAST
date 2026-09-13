"""Shared lab utilities for FAST — Frontier AI Security Training.

Every lab notebook opens with the same two cells:

    <install cell — pip installs this package from GitHub>
    from fast.colab import setup; setup()

The install cell pip-installs `git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast`.
The repo is public, so the clone needs no authentication. Pin `@main` to a release tag
before the program runs, so a commit on Wednesday morning can't change the environment under a
lab running Wednesday afternoon.
"""

__version__ = "0.1.0"
