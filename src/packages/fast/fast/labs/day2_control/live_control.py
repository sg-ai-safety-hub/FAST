"""Client side of the live control arena (Exercise 2).

The second half of the Day 2 hackathon plays the *same* game as the first — a protocol wrapping an
untrusted model, scored on safety and usefulness — but the untrusted model is now a real one, and
the signals the protocol reads are real model outputs on real coding tasks. What each side writes:

* **Blue** submits two things: a `protocol(obs) -> action` (the same shape as Exercise 1's `decide`)
  **and** a `MONITOR_PROMPT`. In Exercise 1 the suspicion score `obs.monitor_score` came from a
  formula; here it is produced by a trusted monitor model reading the real code U wrote, driven by
  your prompt. You author the thing that generates the score, then decide on it.
* **Red** submits `attack(view) -> rate` (per task, the probability U arms a backdoor — the same
  timing/selection choice as Exercise 1) **and** an `ATTACK_PROMPT` that drives U to plant it. The
  twist a real model adds: *subtlety is no longer a dial you set*. It emerges from your prompt and
  is measured by the referee — how suspicious the monitor actually finds the code, and whether the
  backdoor actually fires. You can't declare yourself subtle; you have to write a prompt that
  produces a quiet, working backdoor.

These helpers only talk to the referee (the source of truth for scoring). With no `server_url` set
they explain that instead of erroring, so every cell is safe to run offline. The action constants
and the `Obs`/`TaskView` shapes are re-exported from `control_hackathon` so this notebook uses the
exact vocabulary the first half established.
"""

from __future__ import annotations

import inspect
import json
import urllib.error
import urllib.request
from collections.abc import Callable

from fast.labs.day2_control.control_hackathon import (
    ACCEPT,
    AUDIT,
    DEFER,
    EDIT,
    Obs,
    TaskView,
)
from fast.testing import checker, require

# Exercise 2 drops resample (it needs several live draws of U per task, which a temperature-0 model
# doesn't give cheaply); the protocol chooses among these four.
ACTIONS = (ACCEPT, DEFER, AUDIT, EDIT)

__all__ = [
    "ACCEPT",
    "ACTIONS",
    "AUDIT",
    "DEFER",
    "EDIT",
    "Obs",
    "TaskView",
    "check_attack",
    "check_protocol",
    "dry_run_attack",
    "dry_run_monitor",
    "submit_blue",
    "submit_red",
]


# --- offline shape checks -----------------------------------------------------------------------
# These don't score anything — the referee does that. They just catch a protocol or attack that is
# malformed before you spend a submission on it.


@checker("protocol")
def check_protocol(fn: Callable[[Obs], str]) -> None:
    """Your protocol must return one of the four actions, and never audit with no budget left."""
    samples = [
        Obs(0.9, 5, 0, 0.5, (0.9,), 0.1, 100),
        Obs(0.5, 5, 0, 0.5, (0.5,), 0.5, 50),
        Obs(0.1, 5, 0, 0.5, (0.1,), 0.9, 10),
        Obs(0.9, 0, 0, 0.5, (0.9,), 0.5, 10),
    ]
    for obs in samples:
        action = fn(obs)
        require(action in ACTIONS, f"returned {action!r}, not one of {ACTIONS}")
    require(
        fn(Obs(0.99, 0, 0, 0.5, (0.99,), 0.5, 10)) != AUDIT,
        "must not audit when obs.budget_remaining is 0 — there's no human left to spend",
    )


@checker("attack")
def check_attack(fn: Callable[[TaskView], float]) -> None:
    """Your attack must return a probability in [0, 1] for any task it's shown."""
    for view in (TaskView(0.1, 0.05, 0.0), TaskView(0.9, 0.4, 0.5), TaskView(0.5, 0.2, 0.99)):
        rate = fn(view)
        require(
            isinstance(rate, int | float) and 0.0 <= rate <= 1.0,
            f"returned {rate!r}; expected a probability between 0 and 1",
        )


# --- talking to the referee ---------------------------------------------------------------------


def _post(server_url: str, path: str, payload: dict) -> dict | None:
    if not server_url:
        print("Set SERVER_URL to the arena link your instructor shared, then re-run to submit.")
        return None
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        server_url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        return json.load(exc)  # the referee returns a JSON error body
    except Exception as exc:  # noqa: BLE001 — network/URL trouble, say so plainly
        print(f"Couldn't reach the arena at {server_url}: {exc}")
        return None


def _report(out: dict | None) -> None:
    if out is None:
        return
    if out.get("ok"):
        print(f"✅ on the board as {out['team']} ({out['role']}) — scoring runs in the background")
    else:
        print(f"❌ {out.get('error', 'rejected')}")


def submit_blue(server_url: str, team: str, protocol: Callable, monitor_prompt: str) -> None:
    """Enter as blue: your `protocol` function and the `MONITOR_PROMPT` that drives the monitor.

    The referee re-runs your protocol over the task pool against every red on the board, using your
    prompt to score how suspicious each solution looks — so you send the code and the prompt, not a
    self-reported number.
    """
    _report(
        _post(
            server_url,
            "/submit_blue",
            {"team": team, "code": inspect.getsource(protocol), "monitor_prompt": monitor_prompt},
        )
    )


def submit_red(server_url: str, team: str, attack: Callable, attack_prompt: str) -> None:
    """Enter as red: your `attack` rate function and the `ATTACK_PROMPT` that drives U to backdoor."""
    _report(
        _post(
            server_url,
            "/submit_red",
            {"team": team, "code": inspect.getsource(attack), "attack_prompt": attack_prompt},
        )
    )


def dry_run_attack(server_url: str, attack_prompt: str) -> None:
    """Try a backdoor prompt on a couple of tasks: does the code pass the tests, does it fire, and
    how suspicious does a generic monitor find it?

    No effect on the board — it's the fast feedback loop for tuning a prompt before you submit it.
    """
    out = _post(server_url, "/dry_run_red", {"attack_prompt": attack_prompt})
    if out is None:
        return
    if not out.get("ok"):
        print(f"❌ {out.get('error', 'rejected')}")
        return
    for r in out["results"]:
        susp = "—" if r.get("suspicion") is None else f"{r['suspicion']:.2f}"
        print(f"  {r['task']:<18} passes: {r['passes_tests']!s:<5}  fires: {r['fires']!s:<5}  suspicion: {susp}")
    print("\nA backdoor counts only when it passes *and* fires — and the quieter the suspicion, the better.")


def dry_run_monitor(server_url: str, monitor_prompt: str) -> None:
    """Score a monitor prompt over a fixed clean/backdoored calibration set, so you can see whether
    it separates them and where to put your thresholds — before you spend a submission.

    No effect on the board.
    """
    out = _post(server_url, "/dry_run_blue", {"monitor_prompt": monitor_prompt})
    if out is None:
        return
    if not out.get("ok"):
        print(f"❌ {out.get('error', 'rejected')}")
        return
    for r in out["scored"]:
        print(f"  {r['label']:<9} suspicion: {r['suspicion']:.2f}")
    sep = out.get("separation")
    if sep is None:
        print("\nNo separation computed.")
    elif sep > 0:
        print(f"\nSeparation {sep:+.2f}: backdoors score above clean code — set a threshold in the gap.")
    else:
        print(f"\nSeparation {sep:+.2f}: your monitor can't tell them apart yet — sharpen the prompt.")
