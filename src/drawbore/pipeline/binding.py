"""Pipeline edge data-binding references."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class From:
    """A binding to an upstream agent's output.

    ``From("agent")`` binds the whole output object; ``From("agent.field")``
    binds a single field. The orchestrator uses these to construct each agent's
    isolated input payload.
    """

    ref: str

    @property
    def agent(self) -> str:
        return self.ref.split(".", 1)[0]

    @property
    def field(self) -> str | None:
        return self.ref.split(".", 1)[1] if "." in self.ref else None
