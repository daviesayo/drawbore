import tomllib
from pathlib import Path

from drawbore.errors import DrawboreError, halt_reason_for
from drawbore.mcp import MCPError, MCPAuthError, MCPToolNotFoundError


def test_mcp_errors_are_drawbore_errors():
    assert issubclass(MCPError, DrawboreError)
    assert issubclass(MCPAuthError, MCPError)
    assert issubclass(MCPToolNotFoundError, MCPError)


def test_mcp_error_has_legible_halt_reason():
    # self-declared halt_reason so a failed MCP tool call escalates legibly,
    # not as the generic "agent_error".
    assert halt_reason_for(MCPError("boom")) == "mcp_error"
    assert halt_reason_for(MCPToolNotFoundError("x")) == "mcp_error"


def test_mcp_is_an_optional_extra_not_a_core_dependency():
    repo_root = Path(__file__).resolve().parents[2]  # tests/mcp/ -> tests/ -> repo root
    data = tomllib.loads((repo_root / "pyproject.toml").read_text())
    core = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]
    assert not any("mcp" in dep for dep in core), "mcp must not be a core dependency"
    assert extras["mcp"] == ["google-adk[mcp]"]
