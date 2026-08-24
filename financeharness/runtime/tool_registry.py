"""Tool registry — the tool contract, registration, and per-run session state.

A tool is a :class:`ToolSpec`: a name, a Pydantic request schema, an async
handler, and a tier. **Core** tools are always visible to the model;
**deferred**
tools appear only in a catalog until ``load_tool`` pulls their schema (the
two-tier disclosure / ToolSearch pattern).

Naming: a dotted display path (``data.equity.reference``) maps to an underscored
wire name (``data_equity_reference``) because the OpenAI function-name regex
forbids dots. The registry stores the wire form.

Registries here are explicit objects passed to dispatch — fresh per request,
no global mutable state.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

Tier = Literal["core", "deferred"]
ToolHandler = Callable[[BaseModel], Awaitable["ToolResponse"]]

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MIN_DESCRIPTION_CHARS = 30


class ToolError(Exception):
  """An *expected*, actionable tool failure — no usable data, the provider is

  unavailable, or the input can't be served. The dispatcher renders it as a
  clean
  ``ok=False`` result (the message verbatim, no exception-type prefix); the
  model
  reads the message and adjusts. Distinct from an unexpected bug (which surfaces
  as
  ``handler_exception``). The single convention for "the tool couldn't deliver."
  """


class ToolResponse(BaseModel):
  """Uniform response for every tool: ``markdown`` to the model, ``structured``

  + ``meta`` to the trajectory (and ``structured`` feeds reference-chaining).
  A tool returns this on success and raises :class:`ToolError` when it can't.
  """

  markdown: str
  structured: dict[str, Any] = {}
  meta: dict[str, Any] = {}


@dataclass(frozen=True)
class ToolSpec:
  """A registered tool's contract.

  name:           wire id (underscores, OpenAI-regex-safe)
  display_name:   dotted path for the human-readable catalog
  tier:           "core" (always visible) | "deferred" (catalog-only)
  description:    catalog routing text (>= 30 chars)
  request_schema: Pydantic v2 model for input args
  handler:        async callable receiving a validated request instance
  tags:           freeform routing hints surfaced in the catalog
  in_catalog:     whether the deferred catalog lists this tool individually.
                  False keeps a tool fully callable (``load_tool`` finds it by
                  name) while leaving it out of the prompt — the escape hatch for
                  a source that contributes tools by the dozen and would
                  otherwise price its whole surface into every model call.
  """

  name: str
  display_name: str
  tier: Tier
  description: str
  request_schema: type[BaseModel]
  handler: ToolHandler
  tags: tuple[str, Ellipsis] = field(default_factory=tuple)
  in_catalog: bool = True

  def __post_init__(self):
    if not _NAME_RE.match(self.name):
      raise ValueError(
          f"Tool name {self.name!r} violates {_NAME_RE.pattern} — "
          "use underscores not dots on the wire."
      )
    if len(self.description) < _MIN_DESCRIPTION_CHARS:
      raise ValueError(
          f"Tool description for {self.name} is {len(self.description)} chars; "
          f"minimum {_MIN_DESCRIPTION_CHARS} (catalog routing needs signal)."
      )

  def openai_schema(self):
    """OpenAI function-calling format, passed via the ``tools=`` parameter."""
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.request_schema.model_json_schema(),
        },
    }

  def catalog_line(self):
    """One-line catalog entry for the deferred-tool block in the prompt."""
    tag_suffix = f" Tags: {', '.join(self.tags)}." if self.tags else ""
    return (
        f"- {self.display_name} (`{self.name}`): {self.description}{tag_suffix}"
    )


class ToolRegistry:
  """A set of tool specs. Created fresh per request (no global state)."""

  def __init__(self):
    self._tools: dict[str, ToolSpec] = {}

  def register(self, spec):
    if spec.name in self._tools:
      raise ValueError(f"Tool {spec.name} already registered")
    self._tools[spec.name] = spec

  def clone(self):
    """A shallow copy — same specs, independent dict.

    Lets a run add per-run meta-tools without mutating a shared base registry.
    """
    new = ToolRegistry()
    new._tools = dict(self._tools)
    return new

  def get(self, name):
    """Look up by wire name; fall back to the underscored variant when the

    model emits the dotted display form (a known open-weight-model quirk).
    Collision-safe — no registered name contains a dot.
    """
    spec = self._tools.get(name)
    if spec is None and "." in name:
      spec = self._tools.get(name.replace(".", "_"))
    return spec

  def names(self):
    return sorted(self._tools)

  def core_tools(self):
    return [t for t in self._tools.values() if t.tier == "core"]

  def deferred_tools(self):
    return [t for t in self._tools.values() if t.tier == "deferred"]

  def catalog_text(self):
    """The deferred-tool catalog as one text block for the system prompt.

    Only tools that opted into the catalog appear. The rest stay callable by
    name through ``load_tool`` — something has to stand in for them in the
    prompt (see the MCP hub's per-server index), or the model can't find them.
    """
    deferred = sorted(
        (t for t in self.deferred_tools() if t.in_catalog), key=lambda s: s.name
    )
    return "\n".join(t.catalog_line() for t in deferred)


@dataclass
class ToolSessionState:
  """Per-run state: which deferred tools have been loaded, and the structured

  payload of every successful prior call (the source for ``prev:`` chaining).
  Fresh per agent loop — keeps each run's chain space isolated.
  """

  loaded_deferred: set[str] = field(default_factory=set)
  tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)

  def load(
      self, names, registry
  ):
    """Mark deferred names as loaded; return ``(loaded specs, unknown names)``.

    Best-effort, mirroring ``load_skill``: valid names load even if others are
    unknown (the caller reports the unknown ones so the model self-corrects).
    Core names are silently OK.
    """
    loaded: list[ToolSpec] = []
    unknown: list[str] = []
    for n in names:
      spec = registry.get(n)
      if spec is None:
        unknown.append(n)
        continue
      if spec.tier == "deferred":
        self.loaded_deferred.add(spec.name)
      loaded.append(spec)
    return loaded, unknown

  def visible_schemas(self, registry):
    """Schemas the model can call now: every core tool + every loaded

    deferred tool.
    """
    schemas = [spec.openai_schema() for spec in registry.core_tools()]
    for name in sorted(self.loaded_deferred):
      spec = registry.get(name)
      if spec is not None:
        schemas.append(spec.openai_schema())
    return schemas
