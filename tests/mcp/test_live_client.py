import pytest

from drawbore.mcp import MCPClient
from drawbore.mcp.live import LiveMCPClient


def _real_mcp_sdk_present() -> bool:
    """True only if the *real* MCP SDK is importable, identified by its top-level
    ``ClientSession`` symbol.

    A plain ``find_spec("mcp")`` is not reliable here: under pytest's
    ``--import-mode=importlib`` this repo's own ``tests/mcp`` package (it has an
    ``__init__.py`` but ``tests/`` does not) is imported as the top-level name
    ``mcp``, shadowing the SDK. Probing for ``ClientSession`` distinguishes the
    real SDK from any same-named module on the path — and mirrors exactly the
    contract ``LiveMCPClient._require_mcp`` enforces at connect time.
    """
    try:
        from mcp import ClientSession  # noqa: F401
    except Exception:
        return False
    return True


def test_live_client_implements_the_mcpclient_abc():
    # The class exists and is a concrete MCPClient (constructable without `mcp`,
    # which is only needed at connect-time).
    assert issubclass(LiveMCPClient, MCPClient)


@pytest.mark.skipif(
    _real_mcp_sdk_present(),
    reason="the real mcp SDK is installed; the import-guard path is not exercised",
)
async def test_connect_without_mcp_installed_raises_a_legible_error():
    # In the default test env the real `mcp` SDK is absent: using the live
    # transport must fail closed with an actionable MCPError, not an opaque
    # ImportError.
    from drawbore.mcp import MCPError, MCPServerConfig
    client = LiveMCPClient()
    with pytest.raises(MCPError) as ei:
        await client.connect(MCPServerConfig(name="slack", url="https://slack.com/mcp"))
    assert "drawbore[mcp]" in str(ei.value)   # tells the user how to fix it


@pytest.mark.skipif(
    not _real_mcp_sdk_present(),
    reason="requires the optional `mcp` extra (drawbore[mcp])",
)
def test_live_client_constructs_when_mcp_is_available():
    # When the extra IS installed, the class still constructs cleanly. A real
    # server round-trip needs a live MCP server and is not unit-tested.
    LiveMCPClient()
