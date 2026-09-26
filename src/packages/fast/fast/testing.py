"""Participant-facing check harness.

Labs ship checks so participants can tell whether their implementation is right without
asking an instructor. Checks live in `fast.labs.<exercise>` — installed with the package,
so they're available in Colab but not sitting in the notebook where they'd give the answer
away.

Write checks as *property* assertions (shape, norm, invariants, behaviour on known input)
rather than comparisons against a reference implementation. A check that reads
`require(np.allclose(fn(x), reference(x)))` puts the answer one click away; one that reads
`require(is_unit_norm(fn(x)))` doesn't.
"""

from __future__ import annotations

import functools

__all__ = ["CheckFailed", "checker", "exercise", "require"]


def exercise(fn=None, *, stub: str | None = None):
    """Mark a function as one the participant implements.

        @exercise
        def difference_in_means(harmful, harmless):
            \"\"\"Return the unit-norm direction separating the two sets.\"\"\"
            direction = harmful.mean(axis=0) - harmless.mean(axis=0)
            return direction / np.linalg.norm(direction)

    A no-op at runtime. `tools/build_labs.py` finds it in the AST and replaces the body with
    `raise NotImplementedError` in the participant notebook, keeping the signature and the
    docstring — so the docstring is the spec, and worth writing as one.

    Pass `stub` to leave a scaffold instead. It's read statically from the source, never
    called, so it need not be valid at runtime:

        @exercise(stub="direction = ...\\nreturn ...")

    Most of the time you don't want it. This room can work from a signature and a docstring,
    and a scaffold that pre-shapes the answer removes the part worth doing.
    """

    def decorate(f):
        return f

    return decorate if fn is None else fn


class CheckFailed(AssertionError):
    """A participant's implementation failed a check."""


_passed = 0


def require(condition: bool, message: str) -> None:
    """Assert one property of a participant's implementation.

    `message` is read by someone who is stuck — say what was expected and what was seen,
    not just that something was wrong.
    """
    global _passed
    _passed += 1
    if not condition:
        raise CheckFailed(message)


def checker(label: str):
    """Wrap a check function so it reports pass/fail legibly and still fails loudly.

    Printing is for the participant; raising is for CI, which executes solution notebooks
    and needs a failed check to abort the run.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            global _passed
            _passed = 0
            try:
                fn(*args, **kwargs)
            except NotImplementedError:
                print(f"⬜ {label}: not implemented yet")
                raise CheckFailed(f"{label}: not implemented") from None
            except CheckFailed as exc:
                print(f"❌ {label}: {exc}")
                raise
            except Exception as exc:
                # Someone's code broke rather than being wrong. Let the real traceback
                # through — they need it to debug.
                print(f"❌ {label}: your code raised {type(exc).__name__}: {exc}")
                raise
            print(f"✅ {label}: {_passed} checks passed")

        return wrapper

    return decorate
