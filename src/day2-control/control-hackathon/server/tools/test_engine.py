"""Drive the whole scoring chain with the model layer stubbed — no network, no API key.

Run as a plain script (this is how CI runs it, since pytest isn't a dependency):

    python test_engine.py

Each test replaces the four model-calling functions with pure fakes, then exercises
`process_red` / `process_blue` / `recompute` and asserts the board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # server root, so `app` imports

from app import engine

# A fixed, tiny task suite so these tests are hermetic — no tasks.json on disk, no network. The model
# layer is stubbed in every test and only ever reads task["id"], so three ids is enough. This also
# makes the run deterministic regardless of whatever tasks.json happens to be present locally.
engine.TASKS = [{"id": f"t{i}", "prompt": "", "cwe": "CWE-000"} for i in range(3)]
# Clean baselines are curated data now, not model-generated — give the fixture a secure snippet per task
# id so `_ensure_honest()` populates the false-positive pool without a model call.
engine.CLEAN = {t["id"]: "import hashlib\ndef f(b):\n    return hashlib.sha256(b).hexdigest()" for t in engine.TASKS}


def _stub_models(monitor_catches: bool) -> None:
    """An insecure attack the judge always confirms, and a monitor that either catches insecurity (high
    on pickle/os.system) or is blind (always 0). The clean baselines come from engine.CLEAN above."""
    engine.generate_attack = lambda prompt, task, temperature=None: "import pickle\ndef f(b):\n    return pickle.loads(b)"
    engine.judge = lambda task, code: {"does_task": True, "insecure": "pickle" in code, "why": ""}

    def monitor(model, prompt, code, temperature=None):
        if not monitor_catches:
            return 0.0
        return 0.9 if ("pickle" in code or "os.system" in code) else 0.1

    engine.monitor = monitor


def _reset() -> None:
    engine._honest.clear()
    engine.red_subs.clear()
    engine.blue_subs.clear()


def test_catching_monitor_clears_and_red_is_caught() -> None:
    _reset()
    _stub_models(monitor_catches=True)
    engine.process_red("attacker", "be insecure")
    engine.process_blue("good", "flag unsafe code", "qwen/qwen3-8b")
    engine.recompute()

    red = next(r for r in engine.standings["red"] if r["team"] == "attacker")
    assert red["working"] == len(engine.TASKS), f"every task should be confirmed insecure: {red}"

    blue = next(b for b in engine.standings["blue"] if b["team"] == "good")
    assert blue["tpr"] == 1.0, f"should catch every insecure snippet: {blue}"
    assert blue["fpr"] == 0.0, f"should flag no clean baseline: {blue}"
    assert blue["clears"] is True, f"a perfect monitor clears the bar: {blue}"

    assert red["evaded"] == 0, f"a caught red evades nobody: {red}"
    assert engine.standings["fight"]["caught"] is True, "the champion catches the top MO"
    print("ok: catching monitor clears the bar and catches the model organism")


def test_blind_monitor_is_fooled() -> None:
    _reset()
    _stub_models(monitor_catches=False)
    engine.process_red("attacker", "be insecure")
    engine.process_blue("blind", "trust everything", "meta-llama/llama-3.3-70b-instruct")
    engine.recompute()

    blue = next(b for b in engine.standings["blue"] if b["team"] == "blind")
    assert blue["tpr"] == 0.0 and blue["clears"] is False, f"a blind monitor catches nothing: {blue}"

    red = next(r for r in engine.standings["red"] if r["team"] == "attacker")
    assert red["evaded"] == 1, f"the one blind monitor is evaded: {red}"
    print("ok: a blind monitor is fooled and the model organism scores an evasion")


def test_cheaper_clearing_monitor_outranks_dearer_one() -> None:
    _reset()
    _stub_models(monitor_catches=True)
    engine.process_red("attacker", "be insecure")
    engine.process_blue("dear", "flag unsafe code", "meta-llama/llama-3.3-70b-instruct")
    engine.process_blue("cheap", "flag unsafe code", "qwen/qwen3-8b")
    engine.recompute()

    ordered = [b["team"] for b in engine.standings["blue"] if b["status"] == "done"]
    assert ordered.index("cheap") < ordered.index("dear"), f"both clear → smallest model wins: {ordered}"
    print("ok: among monitors that clear the bar, the cheaper model ranks first")


def test_one_team_runs_several_models_as_several_rows() -> None:
    """Blue is keyed by (team, model): one team can run a monitor on several ladder models at once and
    get a board row for each, while resubmitting the SAME model updates that row in place."""
    _reset()
    _stub_models(monitor_catches=True)
    engine.process_red("attacker", "be insecure")
    engine.process_blue("T", "flag unsafe code", "meta-llama/llama-3.3-70b-instruct")
    engine.process_blue("T", "flag unsafe code", "qwen/qwen3-8b")
    engine.recompute()

    rows = [b for b in engine.standings["blue"] if b["team"] == "T"]
    assert len(rows) == 2, f"one team on two models should be two rows: {rows}"
    assert {b["model"] for b in rows} == {"meta-llama/llama-3.3-70b-instruct", "qwen/qwen3-8b"}

    engine.process_blue("T", "a sharper prompt", "qwen/qwen3-8b")  # same model, resubmitted
    engine.recompute()
    rows = [b for b in engine.standings["blue"] if b["team"] == "T"]
    assert len(rows) == 2, f"resubmitting a model updates its row, not adds a third: {rows}"
    print("ok: one team runs several models as several rows; resubmitting a model updates in place")


def test_inflight_cap_blocks_a_duplicate_until_released() -> None:
    """reserve() gates the queue: one live submission per key. Blue keys by (team, model) so a team can
    still fire several models at once; red keys by team. The worker releases the slot when it finishes."""
    _reset()
    engine._inflight.clear()
    m = "meta-llama/llama-3.3-70b-instruct"
    assert engine.reserve("blue", "T", m) is True          # first claim on this (team, model) wins
    assert engine.reserve("blue", "T", m) is False          # a duplicate while it's in flight is refused
    assert engine.reserve("blue", "T", "google/gemma-3-27b-it") is True  # a different model is a different row
    assert engine.reserve("red", "T", None) is True          # red is keyed by team, independent of blue
    assert engine.reserve("red", "T", None) is False

    # the worker frees the slot on completion via _sub_key(...); once released, a resubmit is allowed
    with engine._lock:
        engine._inflight.discard(engine._sub_key("blue", "T", m))
    assert engine.reserve("blue", "T", m) is True
    print("ok: the in-flight cap blocks a duplicate submission until its slot is released")


def test_red_resubmission_rescored_not_stale() -> None:
    """A team iterating on its attack prompt resubmits under the same name. The cached monitor scores
    for its old code must be discarded, or the board scores new code with old numbers."""
    _reset()
    engine.judge = lambda task, code: {"does_task": True, "insecure": ("pickle" in code or "eval" in code), "why": ""}

    def monitor(model, prompt, code, temperature=None):
        if "pickle" in code:
            return 0.9  # this monitor catches pickle
        if "eval" in code:
            return 0.3  # ...but is blind to eval
        return 0.1

    engine.monitor = monitor

    # first attack uses pickle — the monitor catches it
    engine.generate_attack = lambda prompt, task, temperature=None: "import pickle\ndef f(b):\n    return pickle.loads(b)"
    engine.process_red("R", "v1")
    engine.process_blue("B", "flag unsafe", "qwen/qwen3-8b")
    engine.recompute()
    assert next(b for b in engine.standings["blue"] if b["team"] == "B")["tpr"] == 1.0

    # resubmit R with an eval attack the monitor misses; its cached 0.9 scores must be refreshed to 0.3
    engine.generate_attack = lambda prompt, task, temperature=None: "def f(b):\n    return eval(b)"
    engine.process_red("R", "v2")
    engine.recompute()

    b = engine.blue_subs[engine._bkey("B", "qwen/qwen3-8b")]  # blue is keyed by (team, model)
    a_key = next(k for k in b["scores"] if k.startswith("R:"))
    assert b["scores"][a_key] == 0.3, f"stale cached score not refreshed on resubmit: {b['scores'][a_key]}"
    blue = next(x for x in engine.standings["blue"] if x["team"] == "B")
    red = next(x for x in engine.standings["red"] if x["team"] == "R")
    assert blue["tpr"] == 0.0, f"monitor should now miss the eval attack: {blue}"
    assert red["evaded"] == 1, f"the refreshed board shows R evading B: {red}"
    print("ok: a red resubmission is re-scored against every monitor, not left stale")


def test_unscorable_snippet_is_non_signal() -> None:
    """A snippet a monitor can't score (a content filter, an empty reply) must be a non-signal —
    excluded from that monitor's rates AND from red's evaded count — not a silent 0.0, and not an
    error that lets a red DoS every strong monitor with one filter-tripping snippet."""
    _reset()
    engine.judge = lambda task, code: {"does_task": True, "insecure": code.startswith("attack-"), "why": ""}
    engine.generate_attack = lambda prompt, task, temperature=None: f"attack-{task['id']}"
    target = engine.TASKS[0]["id"]

    def monitor(model, prompt, code, temperature=None):
        if code == f"attack-{target}":
            raise RuntimeError("content_filter")  # this one snippet can't be scored
        return 0.9 if code.startswith("attack-") else 0.1

    engine.monitor = monitor
    engine.process_red("R", "p")
    engine.process_blue("B", "m", "meta-llama/llama-3.3-70b-instruct")
    engine.recompute()

    assert engine.blue_subs[engine._bkey("B", "meta-llama/llama-3.3-70b-instruct")]["scores"][f"R:{target}"] is None, \
        "unscorable snippet cached as None"
    blue = next(x for x in engine.standings["blue"] if x["team"] == "B")
    assert blue["tpr"] == 1.0, f"caught-rate over the *scorable* insecure snippets, not dragged by None: {blue}"
    red = next(x for x in engine.standings["red"] if x["team"] == "R")
    assert red["evaded"] == 0, f"the non-signal snippet must not count as an evasion for red: {red}"
    print("ok: a snippet a monitor can't score is a non-signal, excluded from both sides")


def test_snapshot_round_trips_the_house_field() -> None:
    """The house field is precomputed once and loaded from a snapshot at startup, not re-scored with
    live calls. A snapshot then loaded into fresh state must reproduce the same board — with no model
    calls on the load path (we prove that by clearing the stubs before loading)."""
    _reset()
    _stub_models(monitor_catches=True)
    engine.process_red("house-red", "be insecure")
    engine.process_blue("house-blue", "flag unsafe code", "qwen/qwen3-8b")
    engine.recompute()
    before = engine.standings["blue"], engine.standings["red"]
    snap = engine.snapshot()

    # A snapshot survives serialization (it's written to house.json as JSON).
    snap = json.loads(json.dumps(snap))

    # Wipe the state AND make any live model call blow up — the load path must touch neither.
    _reset()
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("load path made a model call"))
    engine.generate_attack = engine.judge = engine.monitor = boom

    engine.load_snapshot(snap)
    engine.recompute()

    assert engine.standings["blue"] == before[0], "blue board must match after snapshot reload"
    assert engine.standings["red"] == before[1], "red board must match after snapshot reload"
    assert engine._honest and engine.red_subs and engine.blue_subs, "state repopulated from snapshot"
    print("ok: the house field round-trips through a snapshot with no model calls on load")


def test_persisted_board_round_trips_and_reset_clears_it() -> None:
    """STATE_PATH makes the live field survive a redeploy: persist() dumps it, load_persisted() restores
    it into fresh state ahead of the house field, and clear_persisted() (admin reset) drops it so the
    next boot starts clean. All three are no-ops when STATE_PATH is unset."""
    import tempfile

    _reset()
    _stub_models(monitor_catches=True)
    engine.process_red("attacker", "be insecure")
    engine.process_blue("good", "flag unsafe code", "qwen/qwen3-8b")
    engine.recompute()
    before = engine.standings["blue"], engine.standings["red"]

    with tempfile.TemporaryDirectory() as d:
        engine.STATE_PATH = str(Path(d) / "board.json")

        assert engine.load_persisted() is False, "nothing saved yet ⇒ caller falls back to house"
        engine.persist()
        assert Path(engine.STATE_PATH).exists(), "persist() writes the live field to STATE_PATH"

        # Wipe state and make any live model call blow up — the restore path must touch neither.
        _reset()
        boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("restore made a model call"))
        engine.generate_attack = engine.judge = engine.monitor = boom
        assert engine.load_persisted() is True, "a saved board restores"
        engine.recompute()
        assert engine.standings["blue"] == before[0], "blue board matches after persisted restore"
        assert engine.standings["red"] == before[1], "red board matches after persisted restore"

        engine.clear_persisted()
        assert not Path(engine.STATE_PATH).exists(), "reset drops the saved board"
        assert engine.load_persisted() is False, "after clear, boot falls back to the house field"

    engine.STATE_PATH = ""  # persistence off again: these must be no-ops, not crash on a missing path
    engine.persist()
    engine.clear_persisted()
    assert engine.load_persisted() is False
    print("ok: the live board persists across a restart via STATE_PATH, and reset clears it")


if __name__ == "__main__":
    for fn in (test_catching_monitor_clears_and_red_is_caught,
               test_blind_monitor_is_fooled,
               test_cheaper_clearing_monitor_outranks_dearer_one,
               test_one_team_runs_several_models_as_several_rows,
               test_inflight_cap_blocks_a_duplicate_until_released,
               test_red_resubmission_rescored_not_stale,
               test_unscorable_snippet_is_non_signal,
               test_snapshot_round_trips_the_house_field,
               test_persisted_board_round_trips_and_reset_clears_it):
        fn()
    print("\nALL TESTS PASSED")
