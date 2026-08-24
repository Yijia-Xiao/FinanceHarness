"""One-shot research entry point — assemble the harness and run a trajectory.

The canonical "ask a question, get a cited report" helper used by the CLI. The
full tool registry (web + equity/market data + valuation/risk compute, the
latter deferred-but-callable) and the bundled skills are always assembled;
``mode`` (and the legacy ``equity`` flag) select only the system-prompt variant.

Any MCP servers configured in ``configs/mcp.json`` are connected for the life of
the run and their tools join the same registry, so a local data source or a
remote MCP endpoint is one more thing the agent can reach for.
"""

from __future__ import annotations

import json
from pathlib import Path

from financeharness.providers import get_profile
from financeharness.runtime.agent import Agent
from financeharness.tools.research import (
  FetchCache,
  build_equity_research_registry,
  citation_finalizer,
  default_skill_registry,
)


async def run_research(
    question,
    *,
    profile = None,
    reader_profile = None,
    equity = False,
    skill_registry = None,
    history = None,
    clarifications = None,
    mode = None,
    backend = None,
    fetcher = None,
    client = None,
    config = None,
    on_event = None,
    stream_tokens = False,
    grounding_review = None,
    mcp = None,
    mcp_servers = None,
):
  """Run a deep-research trajectory end-to-end and return it (with citations).

  ``profile`` is the orchestrator backbone; ``reader_profile`` is the model
  `visit` uses to read pages (a cloud backbone reads its own pages). ``mode``
  (and the legacy ``equity`` flag) selects the system-prompt variant only — the
  full tool registry is assembled either way. ``grounding_review`` overrides the
  self-grounding pass (default: on in research mode) — a comparison seam.

  ``mcp`` gates the external-MCP integration (default: on whenever servers are
  configured); ``mcp_servers`` overrides the configured list. The returned
  trajectory carries an ``mcp`` key with per-server connection state, so a
  client can show what the run could actually reach.
  """
  from financeharness.runtime.modes import get_mode, resolve_mode

  # Mode selects the prompt variant only — the tool registry is the same full
  # surface for every mode (web visible + equity/valuation deferred-but-callable
  # + skills + load_tool). A consistent toolset means switching mode mid-session
  # never strands the model with history that references a now-missing tool.
  # resolve_mode is the single source shared with the service's run_start frame,
  # so the advertised mode always matches the variant that actually runs.
  variant = get_mode(resolve_mode(mode, equity=equity)).prompt_variant

  profile = profile or get_profile()
  # Pair the reader with the backbone (cloud backbones read with a cloud reader, so
  # a run never depends on the local reader being served). Explicit override wins.
  reader_profile = reader_profile or get_profile(
      profile.reader_profile or "qwen-reader"
  )
  cache = FetchCache()
  registry = build_equity_research_registry(
      cache,
      reader_profile,
      backend=backend,
      fetcher=fetcher,
      client=client,
      config=config,
  )
  skill_registry = skill_registry or default_skill_registry()

  # External MCP servers, connected for the life of this run. Their tools land in
  # the deferred tier alongside the first-party ones, so the model discovers them
  # through the same catalog and loads what it needs. A server that won't connect
  # is reported, not raised.
  hub = await _connect_mcp(registry, mcp, mcp_servers, on_event)

  agent = Agent(
      profile=profile,
      registry=registry,
      config=config,
      client=client,
      finalize=citation_finalizer(cache),
      skill_registry=skill_registry,
      stream_tokens=stream_tokens,
      # One backbone self-grounding pass over the draft — the model rereads its
      # report against the sources it read and grounds any claim it can't support
      # (tool figures it owns stay intact). Research mode only by default: it's a
      # web-research anti-fabrication pass, so auto/analytical skip the extra call.
      # An explicit override wins over the mode default.
      grounding_review=(variant == "research")
      if grounding_review is None
      else grounding_review,
      prompt_variant=variant,
  )
  from financeharness.clarify import format_clarifications

  composed = question + format_clarifications(clarifications)
  try:
    traj = await agent.run(composed, on_event=on_event, history=history)
  finally:
    if hub is not None:
      await hub.aclose()  # stop child processes / close endpoints with the run
  traj["citations"] = [
      {"index": c.index, "url": c.url, "title": c.title}
      for c in cache.citations
  ]
  traj["mcp"] = hub.status() if hub is not None else []
  return traj


async def _connect_mcp(registry, enabled, servers, on_event):
  """Connect the configured MCP servers and register their tools on ``registry``.

  Returns the hub (to close when the run ends), or ``None`` when MCP is off or
  nothing is configured. Never raises: the integration is additive, so a broken
  server config costs the run its extra tools and nothing else.
  """
  from financeharness.mcp import McpHub, load_mcp_servers

  if enabled is False:
    return None
  configs = list(servers) if servers is not None else load_mcp_servers()
  if not configs:
    return None
  try:
    hub = await McpHub.connect(configs, emit=on_event)
  except Exception:  # noqa: BLE001 — additive integration: never fail the run
    return None
  for spec in hub.specs():
    try:
      registry.register(spec)
    except ValueError:  # a name already taken: keep the first-party tool
      continue
  return hub


def save_trajectory(traj, path):
  """Write a trajectory to JSON (parents created)."""
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(traj, indent=2, ensure_ascii=False))
  return p
