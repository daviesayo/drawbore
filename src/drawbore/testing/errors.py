"""Local test-mode error.

Self-declares ``halt_reason`` so a test-mode failure — a
missing/exhausted mock, an unmocked external tool, a missing loop script — escalates
legibly through the normal pipeline halt path without ``drawbore.errors`` importing
``drawbore.testing``."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class TestingError(DrawboreError):
    """A local test run could not proceed without calling the outside world — an
    absent or exhausted model response / loop script / tool mock, or a declared
    external tool with no mock and no ``allow_real_tools`` opt-in. Fail closed: test
    mode never silently falls through to a live provider, database, or MCP server."""

    halt_reason = "testing_error"
