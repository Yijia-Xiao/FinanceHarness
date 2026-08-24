"""The MCP hub — external MCP servers borrowed as harness tools (inbound).

One :class:`McpHub` owns the live connections for a run. Each connected server's
tools arrive as ordinary :class:`ToolSpec`s in the *deferred* tier, so they land
in the prompt's catalog and the model pulls the ones it wants with ``load_tool``
— the same progressive-disclosure path the first-party tools use. Nothing else
in the loop needs to know a tool came from outside the process.

Two ways in, both configured in ``configs/mcp.json``:

  a local process (stdio) — a SQLite / CSV / filesystem / internal-API MCP
  server started as a child. This is the path for private data: the harness gets
  the tool surface, the data never leaves the machine.

  a remote endpoint (http) — a streamable-HTTP MCP server, with headers for auth.

Failure is per-server and never fatal: an unreachable or misbehaving server is
recorded in :meth:`status` and skipped, and the run proceeds with the tools it
does have.
"""

from __future__ import annotations

import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from financeharness.mcp.bridge import (
  call_arguments,
  passthrough_model,
  result_to_response,
)
from financeharness.mcp.config import TOOL_PREFIX, load_mcp_servers
from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec

_MIN_DESCRIPTION_CHARS = 30  # the registry's floor for catalog routing signal
_UNSAFE_NAME = re.compile(r"[^a-zA-Z0-9_]+")
_RESOURCE_LIST_LIMIT = 200
_FIND_TOOLS_LIMIT = 40  # tool descriptions returned per find_tools call


def _slug(text):
  """A wire-name-safe fragment (the registry forbids dots and punctuation)."""
  return _UNSAFE_NAME.sub("_", str(text)).strip("_") or "unnamed"


def _describe(description, *, server, tool):
  """A catalog description that always clears the registry's signal floor.

  The provenance suffix is not decoration: in a trajectory that mixes
  first-party and borrowed tools, which server answered is part of the audit
  trail.
  """
  base = (description or "").strip()
  suffix = f" (via the '{server}' MCP server)"
  if not base:
    base = f"The '{tool}' tool exposed by the '{server}' MCP server."
  out = base + suffix
  if len(out) < _MIN_DESCRIPTION_CHARS:
    out += " Borrowed over the Model Context Protocol."
  return out


class ListResourcesRequest(BaseModel):
  """Input for listing an MCP server's resources."""

  model_config = {"extra": "forbid"}


class FindToolsRequest(BaseModel):
  """Input for searching one MCP server's tool surface."""

  model_config = {"extra": "forbid"}

  query: str | None = Field(
      None,
      description=(
          "Optional filter — a word or fragment matched against tool names and"
          " descriptions (e.g. 'dividend', 'macro', 'options'). Omit to list"
          " every tool."
      ),
  )


class ReadResourceRequest(BaseModel):
  """Input for reading one MCP resource by URI."""

  model_config = {"extra": "forbid"}

  uri: str = Field(
      ...,
      description=(
          "The resource URI to read, exactly as listed by the matching"
          " list_resources tool (e.g. 'file:///data/holdings.csv')."
      ),
  )


@dataclass
class McpServerState:
  """What a run knows about one configured MCP server."""

  name: str
  transport: str
  target: str
  connected: bool = False
  error: str | None = None
  instructions: str | None = None
  catalog_mode: str = "full"  # "full" | "index" (see McpHub._add_server)
  tools: list[str] = field(default_factory=list)
  resources: list[dict[str, Any]] = field(default_factory=list)

  def to_dict(self):
    return {
        "name": self.name,
        "transport": self.transport,
        "target": self.target,
        "connected": self.connected,
        "error": self.error,
        "instructions": self.instructions,
        "catalog_mode": self.catalog_mode,
        "tools": list(self.tools),
        "resource_count": len(self.resources),
    }


class McpHub:
  """Live connections to the configured MCP servers, as harness tool specs."""

  def __init__(self):
    self._stack = AsyncExitStack()
    self._states: list[McpServerState] = []
    self._specs: list[ToolSpec] = []
    self._names: set[str] = set()

  # ----------------------------------------------------------------- lifecycle

  @classmethod
  async def connect(cls, servers=None, *, emit=None):
    """Connect to every enabled server and build the borrowed tool specs.

    Always returns a hub — one that yielded nothing is simply a hub with no
    specs, which is exactly how a run with no MCP config behaves.
    """
    hub = cls()
    configs = load_mcp_servers() if servers is None else list(servers)
    for cfg in configs:
      await hub._add_server(cfg)
    if emit is not None and hub._states:
      emit("mcp", {"servers": hub.status()})
    return hub

  async def aclose(self):
    """Tear down every connection (and any child process) for this run."""
    await self._stack.aclose()

  async def __aenter__(self):
    return self

  async def __aexit__(self, *_exc):
    await self.aclose()
    return False

  # ------------------------------------------------------------------ accessors

  def specs(self):
    """The borrowed tools, ready to register on a run's tool registry."""
    return list(self._specs)

  def status(self):
    """Per-server connection state — for the CLI, the API, and the UI panel."""
    return [s.to_dict() for s in self._states]

  def connected_count(self):
    return sum(1 for s in self._states if s.connected)

  # -------------------------------------------------------------- connecting

  async def _open_client(self, cfg):
    """Enter a client for ``cfg`` on the hub's exit stack and return it."""
    # Imported lazily so `mcp` is only needed when MCP is actually configured.
    from mcp import Client, StdioServerParameters, stdio_client

    if cfg.resolved_transport() == "http":
      import httpx2
      from mcp.client.streamable_http import streamable_http_client

      http_client = None
      if cfg.headers:
        http_client = await self._stack.enter_async_context(
            httpx2.AsyncClient(headers=dict(cfg.headers), timeout=cfg.timeout_s)
        )
      transport = streamable_http_client(cfg.url, http_client=http_client)
    else:
      from mcp.client.stdio import get_default_environment

      params = StdioServerParameters(
          command=cfg.command,
          args=list(cfg.args),
          # The SDK's minimal safe environment plus the config's own additions —
          # a local server usually needs PATH/HOME, and nothing more.
          env={**get_default_environment(), **cfg.env},
          cwd=cfg.cwd,
      )
      transport = stdio_client(params)

    return await self._stack.enter_async_context(
        Client(transport, read_timeout_seconds=cfg.timeout_s)
    )

  async def _add_server(self, raw_cfg):
    """Connect one server, discover its surface, and build its tool specs."""
    cfg = raw_cfg.expanded()
    state = McpServerState(
        name=cfg.name,
        transport=cfg.resolved_transport(),
        # The *unexpanded* target, so a config that references a credential by
        # ${VAR} displays the variable rather than its value. target() redacts a
        # literal one on top of that.
        target=raw_cfg.target(),
    )
    self._states.append(state)
    try:
      client = await self._open_client(cfg)
      tools = (await client.list_tools()).tools
    except Exception as err:  # noqa: BLE001 — one bad server never fails the run
      state.error = f"{type(err).__name__}: {err}"
      return

    state.connected = True
    state.instructions = getattr(client, "instructions", None)

    allow = set(cfg.tools)
    selected = [t for t in tools if not allow or t.name in allow]

    # A server's whole tool surface is priced into every model call if each tool
    # takes a catalog line. Past a threshold, collapse to one line for the server
    # and let the model pull schemas by name — the same progressive disclosure
    # the first-party deferred tier uses, one level up.
    indexed = cfg.catalog == "index" or (
        cfg.catalog == "auto" and len(selected) > cfg.catalog_threshold
    )
    state.catalog_mode = "index" if indexed else "full"

    for tool in selected:
      spec = self._tool_spec(client, cfg, tool, in_catalog=not indexed)
      if spec is not None:
        state.tools.append(tool.name)
        self._specs.append(spec)

    if indexed and state.tools:
      index = self._index_spec(cfg, state, selected)
      if index is not None:
        self._specs.append(index)

    if cfg.resources:
      await self._add_resource_tools(client, cfg, state)

  async def _add_resource_tools(self, client, cfg, state):
    """Bridge a server's resources as a list/read pair, when it has any.

    A data source often exposes no tools at all — just resources (a table, a
    directory, a document set). Without this the harness would connect and find
    nothing to call.
    """
    caps = getattr(client, "server_capabilities", None)
    if caps is not None and getattr(caps, "resources", None) is None:
      return
    try:
      listed = (await client.list_resources()).resources
    except Exception:  # noqa: BLE001 — no resources / not supported: nothing to add
      return
    state.resources = [
        {
            "uri": str(r.uri),
            "name": getattr(r, "name", None),
            "description": getattr(r, "description", None),
            "mime_type": getattr(r, "mime_type", None),
        }
        for r in listed[:_RESOURCE_LIST_LIMIT]
    ]
    if not state.resources:
      return
    for spec in self._resource_specs(client, cfg, state):
      self._specs.append(spec)

  # ------------------------------------------------------------ spec factories

  def _claim(self, name):
    """Reserve a wire name; None when it collides (skipped, never silent)."""
    if name in self._names:
      return None
    self._names.add(name)
    return name

  def _tool_spec(self, client, cfg, tool, *, in_catalog=True):
    """One MCP tool → one deferred harness ToolSpec.

    ``in_catalog=False`` keeps it callable (and loadable by name) but out of the
    prompt; the server's index tool stands in for it.
    """
    wire = self._claim(f"{TOOL_PREFIX}_{_slug(cfg.name)}_{_slug(tool.name)}")
    if wire is None:
      return None
    schema = getattr(tool, "input_schema", None) or {}
    request_schema = passthrough_model(f"{wire}_args", schema)
    label = f"{cfg.name}.{tool.name}"
    remote_name = tool.name

    async def handler(req):
      result = await client.call_tool(remote_name, call_arguments(req))
      markdown, structured = result_to_response(result, tool_label=label)
      return ToolResponse(
          markdown=markdown,
          structured=structured,
          meta={"mcp_server": cfg.name, "mcp_tool": remote_name},
      )

    return ToolSpec(
        name=wire,
        display_name=f"{TOOL_PREFIX}.{cfg.name}.{tool.name}",
        tier="deferred",
        description=_describe(
            getattr(tool, "description", None), server=cfg.name, tool=tool.name
        ),
        request_schema=request_schema,
        handler=handler,
        tags=(TOOL_PREFIX, cfg.name),
        in_catalog=in_catalog,
    )

  def _index_spec(self, cfg, state, tools):
    """One catalog entry standing in for a large server's whole tool surface.

    Its description carries the tool *names* — the discovery key, and cheap —
    while the descriptions stay behind a call. The model narrows with a query,
    then pulls the schemas it wants with ``load_tool``, exactly as it would for a
    first-party deferred tool.
    """
    wire = self._claim(f"{TOOL_PREFIX}_{_slug(cfg.name)}_find_tools")
    if wire is None:
      return None
    catalog = [
        {
            "name": t.name,
            "wire_name": f"{TOOL_PREFIX}_{_slug(cfg.name)}_{_slug(t.name)}",
            "description": (getattr(t, "description", None) or "").strip(),
        }
        for t in tools
    ]
    names = ", ".join(entry["name"] for entry in catalog)

    async def handler(req):
      query = (req.query or "").strip().lower()
      hits = [
          e
          for e in catalog
          if not query
          or query in e["name"].lower()
          or query in e["description"].lower()
      ]
      if not hits:
        raise ToolError(
            f"No tool on the '{cfg.name}' MCP server matches {req.query!r}. "
            f"Available: {names}"
        )
      lines = [
          f"- `{e['wire_name']}` ({e['name']}): {e['description'] or '(no description)'}"
          for e in hits[:_FIND_TOOLS_LIMIT]
      ]
      note = (
          f"\n\n…and {len(hits) - _FIND_TOOLS_LIMIT} more; narrow the query."
          if len(hits) > _FIND_TOOLS_LIMIT
          else ""
      )
      body = "\n".join(lines)
      return ToolResponse(
          markdown=(
              f"{len(hits)} tool(s) on the '{cfg.name}' MCP server"
              + (f" matching {req.query!r}" if query else "")
              + f":\n\n{body}{note}\n\nLoad the ones you need with"
              " `load_tool`, then call them."
          ),
          structured={"tools": hits[:_FIND_TOOLS_LIMIT]},
          meta={"mcp_server": cfg.name, "matched": len(hits)},
      )

    return ToolSpec(
        name=wire,
        display_name=f"{TOOL_PREFIX}.{cfg.name}.find_tools",
        tier="deferred",
        description=(
            f"Search the {len(catalog)} tools the '{cfg.name}' MCP server"
            " exposes, and get the wire names to pass to `load_tool`. Their"
            f" names: {names}."
        ),
        request_schema=FindToolsRequest,
        handler=handler,
        tags=(TOOL_PREFIX, cfg.name, "discovery"),
    )

  def _resource_specs(self, client, cfg, state):
    """The list/read resource tool pair for one server."""
    specs: list[ToolSpec] = []
    server = _slug(cfg.name)
    preview = ", ".join(r["uri"] for r in state.resources[:3])

    list_name = self._claim(f"{TOOL_PREFIX}_{server}_list_resources")
    if list_name is not None:

      async def list_handler(_req):
        rows = ["| URI | Name | Type |", "| --- | --- | --- |"]
        for r in state.resources:
          rows.append(
              f"| `{r['uri']}` | {r.get('name') or ''} | {r.get('mime_type') or ''} |"
          )
        detail = "\n".join(rows)
        return ToolResponse(
            markdown=f"Resources on the '{cfg.name}' MCP server:\n\n{detail}",
            structured={"resources": state.resources},
            meta={"mcp_server": cfg.name},
        )

      specs.append(
          ToolSpec(
              name=list_name,
              display_name=f"{TOOL_PREFIX}.{cfg.name}.list_resources",
              tier="deferred",
              description=(
                  f"List the data resources the '{cfg.name}' MCP server exposes"
                  f" ({len(state.resources)} available, e.g. {preview}). Read one"
                  " with the matching read_resource tool."
              ),
              request_schema=ListResourcesRequest,
              handler=list_handler,
              tags=(TOOL_PREFIX, cfg.name, "resources"),
          )
      )

    read_name = self._claim(f"{TOOL_PREFIX}_{server}_read_resource")
    if read_name is not None:

      async def read_handler(req):
        try:
          result = await client.read_resource(req.uri)
        except Exception as err:  # noqa: BLE001 — actionable: the model retries a URI
          raise ToolError(
              f"could not read '{req.uri}' from the '{cfg.name}' MCP server:"
              f" {type(err).__name__}: {err}"
          ) from err
        parts, payload = [], []
        for item in result.contents:
          text = getattr(item, "text", None)
          if text:
            parts.append(str(text))
            payload.append({"uri": str(item.uri), "text": str(text)})
          else:
            mime = getattr(item, "mime_type", None) or "application/octet-stream"
            parts.append(f"({item.uri} — binary resource, {mime}; not inlined)")
        if not parts:
          raise ToolError(f"'{req.uri}' returned no readable content.")
        return ToolResponse(
            markdown="\n\n".join(parts),
            structured={"uri": req.uri, "contents": payload},
            meta={"mcp_server": cfg.name, "mcp_resource": req.uri},
        )

      specs.append(
          ToolSpec(
              name=read_name,
              display_name=f"{TOOL_PREFIX}.{cfg.name}.read_resource",
              tier="deferred",
              description=(
                  f"Read one resource from the '{cfg.name}' MCP server by URI —"
                  " the contents of a file, table, or document it exposes. List"
                  " the available URIs first."
              ),
              request_schema=ReadResourceRequest,
              handler=read_handler,
              tags=(TOOL_PREFIX, cfg.name, "resources"),
          )
      )
    return specs
