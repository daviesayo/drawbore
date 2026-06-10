"""Regulator-readable resume ledger.

Every run — fresh or resumed — produces a ledger stating what the runtime
observed about prior state for the run id: whether a resume happened, the
topology verdict, and a per-step disposition with its seal verdict. On a
refused resume the ledger asserts ONLY the steps that were
checkpoint-completed; it never claims execution state it did not observe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from drawbore.state.step_seal import field_label

Disposition = Literal["executed", "restored", "skipped", "refused"]
SealVerdict = Literal["recorded", "verified", "drifted", "missing", "none"]
TopologyVerdict = Literal["untracked", "recorded", "verified", "drifted"]


class ResumeLedgerEntry(BaseModel):
    """One step's resume disposition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    agent: str
    disposition: Disposition
    seal: SealVerdict
    drifted_fields: tuple[str, ...] = ()


class ResumeLedger(BaseModel):
    """The run's resume-admission evidence, attached to every ``RunResult``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    resumed: bool
    topology: TopologyVerdict
    entries: tuple[ResumeLedgerEntry, ...]

    def legible(self) -> str:
        """Render the ledger as prose a non-engineer can read."""
        lines: list[str] = []
        if not self.resumed:
            lines.append(
                f"Run '{self.run_id}': fresh run, no prior checkpointed state."
            )
        else:
            lines.append(f"Run '{self.run_id}': resumed from checkpointed state.")
        if self.topology == "drifted":
            lines.append(
                "Resume refused before execution: the pipeline's topology "
                "changed since the checkpoint was taken."
            )
        refused = [e for e in self.entries if e.disposition == "refused"]
        for e in refused:
            if e.seal == "missing":
                lines.append(
                    f"Step {e.index} '{e.agent}': resume refused — no seal was "
                    "stored for this completed step, so its semantics cannot "
                    "be verified."
                )
            else:
                fields = ", ".join(field_label(f) for f in e.drifted_fields)
                lines.append(
                    f"Step {e.index} '{e.agent}': resume refused — changed "
                    f"since checkpoint ({fields})."
                )
        verified = sum(
            1 for e in self.entries
            if e.disposition == "restored" and e.seal == "verified"
        )
        if verified:
            lines.append(
                f"{verified} completed step(s) verified against their seals "
                "and restored without re-execution."
            )
        executed = sum(1 for e in self.entries if e.disposition == "executed")
        if executed:
            lines.append(f"{executed} step(s) executed in this attempt.")
        skipped = sum(1 for e in self.entries if e.disposition == "skipped")
        if skipped:
            lines.append(f"{skipped} step(s) skipped by branching.")
        return "\n".join(lines)


class ResumeLedgerBuilder:
    """Mutable accumulator threaded through a pipeline run.

    Constructed by ``Pipeline.run`` and passed into the run loop like the
    audit recorder; built and attached to the result in one place.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._resumed = False
        self._topology: TopologyVerdict = "untracked"
        self._entries: list[ResumeLedgerEntry] = []

    def set_resumed(self) -> None:
        self._resumed = True

    def set_topology(self, verdict: TopologyVerdict) -> None:
        self._topology = verdict

    def executed(self, index: int, agent: str, *, sealed: bool) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=agent, disposition="executed",
            seal="recorded" if sealed else "none",
        ))

    def restored(self, index: int, agent: str, *, verified: bool) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=agent, disposition="restored",
            seal="verified" if verified else "none",
        ))

    def refused(
        self, index: int, agent: str, *, drifted_fields: tuple[str, ...]
    ) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=agent, disposition="refused",
            seal="drifted" if drifted_fields else "missing",
            drifted_fields=drifted_fields,
        ))

    def skipped(self, index: int, agent: str) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=agent, disposition="skipped", seal="none",
        ))

    def restored_join(self, index: int, name: str) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=name, disposition="restored", seal="none",
        ))

    def executed_join(self, index: int, name: str) -> None:
        self._entries.append(ResumeLedgerEntry(
            index=index, agent=name, disposition="executed", seal="none",
        ))

    def build(self) -> ResumeLedger:
        return ResumeLedger(
            run_id=self._run_id,
            resumed=self._resumed,
            topology=self._topology,
            entries=tuple(self._entries),
        )
