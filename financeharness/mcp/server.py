"""FinanceHarness as an MCP server — the harness exposed to any MCP client (outbound).

Point Claude Desktop, an IDE, or any MCP host at ``fh mcp`` and the finance
surface becomes theirs:

  tools     every first-party harness tool (equity + market data, valuation,
            risk, web search/visit, exact arithmetic), each with its real schema,
            plus one ``deep_research`` tool that runs a full cited trajectory.
  resources the tool catalog and every bundled skill's recipe, so a client can
            attach the workflow it wants as context.
  prompts   one per skill — ``dcf-valuation``, ``equity-deep-dive``, … — ready to
            drop into a conversation.

Calls route through the harness dispatcher, so an MCP client gets the same
argument coercion, the same actionable error text, and the same ``prev:<call_id>``
reference chaining a first-party run gets: a price series can feed a correlation
without the client ever holding the numbers.

The exposed tools are bridged dynamically from the registry — a tool added to the
harness appears here with no edit to this file.
"""

from __future__ import annotations

import itertools
from typing import Any

from financeharness.mcp.bridge import annotated_signature
from financeharness.providers import get_profile
from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolSessionState
from financeharness.tools.research import (
  FetchCache,
  build_equity_research_registry,
  default_skill_registry,
)

SERVER_NAME = "financeharness"
SERVER_INSTRUCTIONS = """\
FinanceHarness exposes grounded financial research over MCP.

Data and compute tools return real figures (company and market data via yfinance;
valuation and risk computed from what you pass them) — prefer them over recalling
numbers. `search` and `visit` read the web and build a citable bibliography.
`deep_research` runs the full autonomous loop and returns a cited report; it takes
minutes, so use it for a whole question rather than a single lookup.

Every tool result ends with a call_id. Feed a prior result into a later call by
reference — `prev:<call_id>.<path>`, e.g. `prev:call_3.bars[*].close` — instead of
retyping a series.

The bundled skills (available as prompts and as resources) are the recommended
workflows: ticker-snapshot, dcf-valuation, relative-valuation, consensus-check,
equity-deep-dive.
"""

# Loop-internal meta-tools: they steer the harness's own agent loop and mean
# nothing to an outside client, which drives its own loop.
_INTERNAL_TOOLS = frozenset({"update_plan", "load_tool", "load_skill"})

_SKILL_URI = "financeharness://skills/{name}"
_CATALOG_URI = "financeharness://tools"


class _Session:
  """One MCP server process's shared harness state.

  The registry, the fetch/citation cache, and the chaining store live for the
  process, so `compose_citations` sees every page this client read and a
  `prev:` reference resolves across calls — the same continuity a single
  harness run has.
  """

  def __init__(self, *, reader_profile=None, backbone=None):
    self.backbone = backbone or get_profile()
    self.reader = reader_profile or get_profile(
        self.backbone.reader_profile or "qwen-reader"
    )
    self.cache = FetchCache()
    self.registry = build_equity_research_registry(self.cache, self.reader)
    self.state = ToolSessionState()
    self.skills = default_skill_registry()
    self._ids = itertools.count(1)
    self.skipped: list[str] = []  # tools that couldn't be bridged (see build_server)
    # Deferred tools are a prompt-budget device for the harness's own loop; an
    # MCP client sees the full surface, so pre-load them all.
    self.state.load(self.registry.names(), self.registry)

  def next_call_id(self):
    return f"call_{next(self._ids)}"

  def exposed_specs(self):
    """The harness tools worth exposing, in catalog order."""
    return sorted(
        (
            self.registry.get(name)
            for name in self.registry.names()
            if name not in _INTERNAL_TOOLS
        ),
        key=lambda s: s.display_name,
    )


def _bridge_tool(server, session, spec):
  """Register one harness ToolSpec on ``server`` with its real schema.

  The MCP SDK derives a tool's schema from the Python signature, so we
  synthesize a function whose parameters mirror the spec's Pydantic request
  model (types, defaults and descriptions intact) and route the call through the
  harness dispatcher.
  """
  names, annotations, defaults = annotated_signature(spec.request_schema)
  if not all(n.isidentifier() for n in names):
    return False  # a field name that can't be a Python parameter: skip, don't guess

  async def invoke(arguments):
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError

    args = {k: v for k, v in arguments.items() if v is not None}
    result = await dispatch(
        spec.name,
        args,
        registry=session.registry,
        session_state=session.state,
        call_id=session.next_call_id(),
    )
    if not result.ok:
      # The dispatcher's message is already the actionable one; flag it as an
      # error so the client's model sees a failure rather than prose.
      raise McpToolError(result.markdown)
    return result.markdown

  # annotated_signature orders required parameters first, so the optional tail is
  # exactly the last len(defaults) names — a legal Python signature either way.
  split = len(names) - len(defaults)
  signature = ", ".join([*names[:split], *(f"{n}=None" for n in names[split:])])
  payload = ", ".join(f"{n!r}: {n}" for n in names)
  source = f"async def {spec.name}({signature}):\n    return await _invoke({{{payload}}})\n"
  namespace: dict[str, Any] = {"_invoke": invoke}
  exec(source, namespace)  # noqa: S102 — the signature is derived from our own schemas
  fn = namespace[spec.name]
  fn.__annotations__ = annotations
  if defaults:
    fn.__defaults__ = defaults
  fn.__doc__ = spec.description
  server.add_tool(fn, name=spec.name, description=spec.description)
  return True


def _add_deep_research(server, session):
  """The whole harness as one tool: question in, cited report out."""

  async def deep_research(question, mode="research", profile=None):
    from financeharness.research import run_research

    traj = await run_research(
        question,
        profile=get_profile(profile) if profile else session.backbone,
        mode=mode,
        mcp=False,  # this *is* the MCP surface; don't recurse into other servers
    )
    report = traj.get("prediction") or "(no answer produced)"
    sources = traj.get("citations") or []
    if sources:
      lines = "\n".join(f"[{c['index']}] {c['title']} — {c['url']}" for c in sources)
      report += f"\n\n---\n{len(sources)} sources read:\n{lines}"
    return report

  deep_research.__annotations__ = {
      "question": str,
      "mode": str,
      "profile": str | None,
      "return": str,
  }
  deep_research.__doc__ = (
      "Run the full autonomous financial deep-research loop and return a cited"
      " report. The agent plans, searches and reads the web, pulls company and"
      " market data, computes valuation/risk from that data, and writes the"
      " report with a bibliography. Takes minutes — use it for a whole question"
      " (e.g. 'Is NVDA overvalued at today's price?'), not a single lookup."
      " mode: research (web-first, the default), analytical (numbers-first), or"
      " auto. profile: a backbone name (gemini, gpt, openrouter, qwen); omit for"
      " the configured default."
  )
  server.add_tool(deep_research, name="deep_research")


def _add_skill_surface(server, session):
  """Skills as prompts (drop the workflow into a chat) and as resources."""
  from mcp.server.mcpserver.prompts.base import Prompt
  from mcp.server.mcpserver.resources.types import TextResource

  for skill in session.skills.all():
    body = skill.body
    uri = _SKILL_URI.format(name=skill.name)
    server.add_resource(
        TextResource(
            uri=uri,
            name=f"skill: {skill.name}",
            description=skill.description,
            mime_type="text/markdown",
            text=body,
        )
    )

    def make_prompt(text=body, description=skill.description):
      def prompt():
        return text

      prompt.__doc__ = description
      return prompt

    server.add_prompt(
        Prompt.from_function(
            make_prompt(), name=skill.name, description=skill.description
        )
    )


def _add_catalog_resource(server, session):
  """A single readable index of the exposed tools."""
  from mcp.server.mcpserver.resources.types import TextResource

  lines = [
      f"- `{spec.name}` ({spec.display_name}): {spec.description}"
      for spec in session.exposed_specs()
  ]
  server.add_resource(
      TextResource(
          uri=_CATALOG_URI,
          name="FinanceHarness tool catalog",
          description="Every harness tool exposed over MCP, with its routing description.",
          mime_type="text/markdown",
          text="# FinanceHarness tools\n\n" + "\n".join(lines),
      )
  )


def build_server(*, backbone=None, reader_profile=None, include_deep_research=True):
  """Assemble the MCP server: bridged tools, deep_research, skills, catalog."""
  from mcp.server import MCPServer

  from financeharness import __version__

  session = _Session(backbone=backbone, reader_profile=reader_profile)
  server = MCPServer(
      name=SERVER_NAME,
      title="FinanceHarness",
      version=__version__,
      instructions=SERVER_INSTRUCTIONS,
      website_url="https://github.com/Yijia-Xiao/FinanceHarness",
  )
  for spec in session.exposed_specs():
    if not _bridge_tool(server, session, spec):
      # A field name that can't be a Python parameter. No current tool hits this,
      # but a silent drop would read as "everything is exposed" when it isn't.
      session.skipped.append(spec.name)
  if include_deep_research:
    _add_deep_research(server, session)
  _add_skill_surface(server, session)
  _add_catalog_resource(server, session)
  return server, session


def describe_surface():
  """What ``fh mcp`` would expose — for the CLI's ``--list`` (no client needed)."""
  server, session = build_server()
  del server
  return {
      "name": SERVER_NAME,
      "skipped": list(session.skipped),
      "tools": [
          {"name": spec.name, "description": spec.description}
          for spec in session.exposed_specs()
      ]
      + [{"name": "deep_research", "description": "Full cited research trajectory."}],
      "skills": [s.name for s in session.skills.all()],
      "backbone": session.backbone.name,
      "reader": session.reader.name,
  }
