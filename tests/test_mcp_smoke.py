import pytest

def test_mcp_tools_present_when_enabled():
    """
    Stage 7 MCP smoke test.

    This test asserts that:
    - MCP is enabled in configuration
    - An MCP server is running and reachable
    - MCP tools are discovered and injected into the global TOOLS registry

    NOTE:
    - Requires MCP server running, e.g.:
      uvicorn mcp_server_math.server:app --port 8080
    - This is an integration smoke test, not a unit test.
    """
    from tools import TOOLS

    assert "mcp.math.add_numbers" in TOOLS
    assert "mcp.math.subtract_numbers" in TOOLS