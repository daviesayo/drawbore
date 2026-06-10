"""Resume ledger model + legibility tests."""

from drawbore.state.resume_ledger import (
    ResumeLedger,
    ResumeLedgerBuilder,
    ResumeLedgerEntry,
)


def test_builder_fresh_run():
    b = ResumeLedgerBuilder(run_id="r1")
    b.set_topology("recorded")
    b.executed(0, "retriever", sealed=True)
    b.executed(1, "writer", sealed=True)
    ledger = b.build()
    assert ledger.run_id == "r1"
    assert ledger.resumed is False
    assert ledger.topology == "recorded"
    assert [e.disposition for e in ledger.entries] == ["executed", "executed"]
    assert [e.seal for e in ledger.entries] == ["recorded", "recorded"]


def test_builder_no_store_run():
    b = ResumeLedgerBuilder(run_id="r1")
    # topology defaults to "untracked" when never set
    b.executed(0, "retriever", sealed=False)
    ledger = b.build()
    assert ledger.topology == "untracked"
    assert ledger.entries[0].seal == "none"


def test_builder_clean_resume():
    b = ResumeLedgerBuilder(run_id="r1")
    b.set_resumed()
    b.set_topology("verified")
    b.restored(0, "retriever", verified=True)
    b.executed(1, "writer", sealed=True)
    ledger = b.build()
    assert ledger.resumed is True
    assert ledger.entries[0].disposition == "restored"
    assert ledger.entries[0].seal == "verified"


def test_builder_refused_resume():
    b = ResumeLedgerBuilder(run_id="r1")
    b.set_resumed()
    b.set_topology("verified")
    b.restored(0, "retriever", verified=True)
    b.refused(1, "writer", drifted_fields=("model",))
    b.refused(2, "scorer", drifted_fields=())  # missing seal
    ledger = b.build()
    assert ledger.entries[1].disposition == "refused"
    assert ledger.entries[1].seal == "drifted"
    assert ledger.entries[1].drifted_fields == ("model",)
    assert ledger.entries[2].seal == "missing"


def test_builder_skipped_and_join():
    b = ResumeLedgerBuilder(run_id="r1")
    b.skipped(0, "optional_step")
    b.restored_join(1, "merge")
    b.executed_join(2, "merge2")
    ledger = b.build()
    assert ledger.entries[0].disposition == "skipped"
    assert ledger.entries[0].seal == "none"
    assert ledger.entries[1].disposition == "restored"
    assert ledger.entries[1].seal == "none"
    assert ledger.entries[2].disposition == "executed"
    assert ledger.entries[2].seal == "none"


def test_ledger_is_frozen():
    entry = ResumeLedgerEntry(
        index=0, agent="a", disposition="executed", seal="recorded"
    )
    ledger = ResumeLedger(
        run_id="r1", resumed=False, topology="recorded", entries=(entry,)
    )
    import pydantic
    try:
        ledger.resumed = True
        raised = False
    except pydantic.ValidationError:
        raised = True
    assert raised


def test_legible_refusal_names_step_and_friendly_field():
    b = ResumeLedgerBuilder(run_id="order-A-1")
    b.set_resumed()
    b.set_topology("verified")
    b.restored(0, "retriever", verified=True)
    b.refused(1, "confirmation_writer", drifted_fields=("model",))
    text = b.build().legible()
    assert "order-A-1" in text
    assert "confirmation_writer" in text
    assert "model" in text
    assert "refused" in text.lower()


def test_legible_friendly_label_for_fingerprint_field():
    b = ResumeLedgerBuilder(run_id="r1")
    b.set_resumed()
    b.set_topology("verified")
    b.refused(0, "writer", drifted_fields=("instructions_fingerprint",))
    text = b.build().legible()
    assert "instructions" in text
    assert "instructions_fingerprint" not in text


def test_legible_fresh_run_reads_clean():
    b = ResumeLedgerBuilder(run_id="r1")
    b.set_topology("recorded")
    b.executed(0, "retriever", sealed=True)
    text = b.build().legible()
    assert "fresh" in text.lower() or "no prior" in text.lower()
