"""Cross-restart resume with the durable file-backed checkpoint store.

A pipeline halts partway, the process "restarts" (a brand-new store instance
reads the same directory), and the resume completes — restoring completed
steps without re-execution, with no spurious ``resume_drift``. A genuine
contract change still refuses, proving the durable store preserves the
refuse-on-drift safety control across restarts.
"""

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.state import FileCheckpointStore


class Doc(BaseModel):
    text: str


class Fetched(BaseModel):
    content: str


class Verdict(BaseModel):
    ok: bool


CALLS: dict[str, int] = {}


def _agents(*, fail_writer: bool, fetcher_version: str = "1.0.0"):
    """fetcher -> scorer -> writer; writer fails on the first attempt only."""

    @agent(input=Doc, output=Fetched, name="fetcher", version=fetcher_version)
    async def fetcher(payload: Doc) -> Fetched:
        CALLS["fetcher"] = CALLS.get("fetcher", 0) + 1
        return Fetched(content=payload.text)

    @agent(input=Fetched, output=Verdict, name="scorer", version="1.0.0")
    async def scorer(payload: Fetched) -> Verdict:
        CALLS["scorer"] = CALLS.get("scorer", 0) + 1
        return Verdict(ok=payload.content == "ok")

    @agent(input=Verdict, output=Verdict, name="writer", version="1.0.0")
    async def writer(payload: Verdict) -> Verdict:
        CALLS["writer"] = CALLS.get("writer", 0) + 1
        if fail_writer:
            raise RuntimeError("writer exploded")
        return Verdict(ok=payload.ok)

    return fetcher, scorer, writer


def _pipeline(fetcher, scorer, writer) -> Pipeline:
    p = Pipeline("durable-resume")
    p.add(fetcher)
    p.add(scorer)
    p.add(writer)
    return p


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


@pytest.mark.asyncio
async def test_resume_across_process_restart_completes(tmp_path):
    # --- attempt 1: writer explodes; fetcher + scorer checkpointed to disk ---
    store1 = FileCheckpointStore(tmp_path)
    first = await _pipeline(*_agents(fail_writer=True)).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store1
    )
    assert first.status == "halted"
    assert CALLS == {"fetcher": 1, "scorer": 1, "writer": 1}

    # --- "process restart": a brand-new store instance, brand-new agents ---
    store2 = FileCheckpointStore(tmp_path)
    second = await _pipeline(*_agents(fail_writer=False)).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store2
    )

    assert second.status == "completed"
    assert second.halt_code is None
    # fetcher + scorer restored from disk (NOT re-run); only writer re-ran.
    assert CALLS == {"fetcher": 1, "scorer": 1, "writer": 2}

    ledger = second.resume_ledger
    assert ledger is not None
    assert ledger.resumed is True
    assert ledger.topology == "verified"
    assert ledger.entries[0].disposition == "restored"
    assert ledger.entries[0].seal == "verified"
    assert ledger.entries[1].disposition == "restored"
    assert ledger.entries[1].seal == "verified"
    assert ledger.entries[2].disposition == "executed"

    # The restored output is the real typed model with its fields intact.
    assert second.outputs["scorer"] == Verdict(ok=True)


@pytest.mark.asyncio
async def test_genuine_drift_still_refuses_across_restart(tmp_path):
    store1 = FileCheckpointStore(tmp_path)
    first = await _pipeline(*_agents(fail_writer=True)).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store1
    )
    assert first.status == "halted"

    # Restart, but the (completed) fetcher's declared version drifted.
    store2 = FileCheckpointStore(tmp_path)
    second = await _pipeline(*_agents(fail_writer=False, fetcher_version="9.9.9")).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store2
    )
    assert second.status == "halted"
    assert second.halt_code == "resume_drift"
    assert "fetcher" in second.reason
    assert "version" in second.reason
    # Nothing re-ran in the refused attempt.
    assert CALLS == {"fetcher": 1, "scorer": 1, "writer": 1}
    assert second.steps_run == 0


@pytest.mark.asyncio
async def test_torn_write_does_not_corrupt_resume(tmp_path):
    """A stray temp file from a crash mid-write must not break the resume."""
    store1 = FileCheckpointStore(tmp_path)
    first = await _pipeline(*_agents(fail_writer=True)).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store1
    )
    assert first.status == "halted"

    # Drop a half-written temp file into the run directory.
    run_dir = store1._run_dir("run-A")
    (run_dir / "step-00002.json.tmp-zzz").write_text("{ broken")

    store2 = FileCheckpointStore(tmp_path)
    second = await _pipeline(*_agents(fail_writer=False)).run(
        Doc(text="ok"), run_id="run-A", checkpoints=store2
    )
    assert second.status == "completed"
    assert CALLS == {"fetcher": 1, "scorer": 1, "writer": 2}
