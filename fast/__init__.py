"""Shared lab utilities for FAST — Frontier AI Security Training.

Every lab notebook opens with the same two cells:

    !pip install -q git+https://github.com/le0kar0ub1/FAST.git@main
    from fast.colab import setup; setup()

Pin `@main` to a release tag before the program runs, so a commit on Wednesday
morning can't change the environment under a lab running Wednesday afternoon.
"""

__version__ = "0.1.0"
