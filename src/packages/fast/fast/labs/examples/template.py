"""Fixtures and checks for the worked example in `examples/template/`.

Copy this file to `fast/labs/<day>/<exercise>.py` when starting a real lab. It exists to be read
alongside `examples/template/lab.py` — together they show the whole pattern, and CI executes
them on every push, so if the machinery breaks you find out here first.

Deliberately numpy-only and CPU-only: the example has to run in CI, which has no GPU.
"""

from __future__ import annotations

import numpy as np

from fast.testing import checker, require

DIM = 16


def activation_pairs(n: int = 64, dim: int = DIM, seed: int = 0):
    """Synthetic 'harmful' and 'harmless' activations separated along one known axis.

    Stands in for the real thing in `day3-weights/abliteration`, where these come from
    running a model on paired prompts.
    """
    rng = np.random.default_rng(seed)
    harmless = rng.normal(size=(n, dim))
    offset = np.zeros(dim)
    offset[0] = 2.0
    return harmless + offset, harmless


@checker("difference_in_means")
def check_difference_in_means(fn) -> None:
    harmful, harmless = activation_pairs()
    result = fn(harmful, harmless)

    shape = getattr(result, "shape", None)
    require(shape == (DIM,), f"expected shape ({DIM},), got {shape or type(result).__name__}")

    norm = float(np.linalg.norm(result))
    require(np.isclose(norm, 1.0, atol=1e-6), f"direction should be unit norm, got {norm:.4f}")

    axis = np.zeros(DIM)
    axis[0] = 1.0
    alignment = abs(float(result @ axis))
    require(
        alignment > 0.9,
        f"direction doesn't separate the two sets (alignment {alignment:.2f}, expected > 0.9) — "
        "are you taking the mean over the right axis?",
    )

    # Sign convention matters downstream: the direction should point harmful-ward.
    require(float(result @ axis) > 0, "direction points the wrong way — check the subtraction order")


@checker("project_out")
def check_project_out(fn) -> None:
    rng = np.random.default_rng(1)
    direction = np.zeros(DIM)
    direction[0] = 1.0
    activations = rng.normal(size=(8, DIM))

    result = fn(activations, direction)
    require(
        getattr(result, "shape", None) == activations.shape,
        f"expected shape {activations.shape}, got {getattr(result, 'shape', type(result).__name__)}",
    )

    residual = np.abs(result @ direction).max()
    require(
        residual < 1e-6,
        f"the direction survives projection (max component {residual:.2e}) — "
        "the result should have no remaining component along it",
    )

    orthogonal = activations - np.outer(activations @ direction, direction)
    require(
        np.allclose(result, orthogonal, atol=1e-6),
        "everything orthogonal to the direction should be left untouched",
    )
