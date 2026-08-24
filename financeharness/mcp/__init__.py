"""Model Context Protocol integration — both directions.

inbound  (:mod:`financeharness.mcp.hub`)
    External MCP servers become harness tools. A local stdio server (SQLite, a
    CSV directory, an internal API) or a remote streamable-HTTP endpoint is
    configured in ``configs/mcp.json``; its tools and resources arrive in the
    deferred tier and the agent loads what it needs. Private data stays local —
    the harness gets the tool surface, not a copy of the data.

outbound (:mod:`financeharness.mcp.server`)
    The harness becomes an MCP server (``fh mcp``). Any MCP client — Claude
    Desktop, an IDE, another agent — gets the finance toolkit, the bundled skills
    as prompts, and a ``deep_research`` tool that returns a cited report.

Note on names: this package is ``financeharness.mcp``; the SDK is the top-level
``mcp``. Absolute imports keep them apart.
"""

from financeharness.mcp.config import (
    McpServerConfig,
    config_path,
    load_mcp_servers,
    mcp_disabled,
)
from financeharness.mcp.hub import McpHub

__all__ = [
    "McpHub",
    "McpServerConfig",
    "config_path",
    "load_mcp_servers",
    "mcp_disabled",
]
