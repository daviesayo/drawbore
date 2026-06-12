"""The StepExecutor result contract.

The scheduler (`_run_inner`) decides WHICH nodes run; the executor runs ONE agent
and returns one of these. The executor owns no cross-step state — everything the
scheduler needs to record the step is carried in `StepAudit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    # Deferred: drawbore.llm transitively imports litellm. Annotation-only use is
    # safe under `from __future__ import annotations`; a future
    # get_type_hints(StepAudit) call in a cold-import context would NameError.
    from drawbore.llm import ModelAudit, TokenUsage


@dataclass(frozen=True)
class StepAudit:
    """Everything the recorder needs about one executed step, so the scheduler
    never re-reads shared state (e.g. the proxy log). `tool_calls` is the rendered
    delta this step produced."""

    input_hash: str | None
    tool_calls: tuple[str, ...] = ()
    evidence_summary: str | None = None
    model_audit: ModelAudit | None = None
    model_turns: int = 0
    reprompts: int = 0
    tokens: "TokenUsage | None" = None
    cost: float | None = None


@dataclass(frozen=True)
class Ok:
    """The agent produced a schema-valid output."""

    output: BaseModel
    audit: StepAudit
    validated_input: BaseModel | None = None  # the step's validated input; scheduler passes it as escalation `received`


@dataclass(frozen=True)
class Halt:
    """The step failed (schema/tool/agent error). The scheduler turns this into the
    run's halt-and-escalate. `audit` lets it record a FAILED step when work was done.

    `code` is the stable halt token (one of drawbore.errors.HALT_CODES), passed
    explicitly — never parsed back out of `reason`."""

    reason: str
    received: Any
    attempted: Any
    audit: StepAudit
    code: str = "agent_error"


Outcome = Union[Ok, Halt]
