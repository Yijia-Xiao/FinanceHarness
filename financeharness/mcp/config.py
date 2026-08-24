"""MCP server configuration — which external MCP servers this harness connects to.

Two transports, distinguished by what you point at:

  stdio — a local process (``command`` + ``args``). This is how a local data
          source joins the harness: a SQLite/CSV/filesystem MCP server runs as a
          child process and speaks MCP over its stdin/stdout. Nothing leaves the
          machine.
  http  — a remote (or local) MCP endpoint over streamable HTTP (``url``).

Precedence mirrors ``runtime/config.py`` and ``providers/profiles.py``: built-in
defaults (none) < the JSON file (``configs/mcp.json``, or ``FH_MCP_CONFIG``) <
env. ``FH_MCP_DISABLE=1`` turns the whole integration off without editing files.
Never raises on a missing or malformed file — a bad config yields no servers
rather than a broken run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "mcp.json"

# Wire-name prefix for every bridged MCP tool, so a glance at a trajectory tells
# you which figures came from outside the harness.
TOOL_PREFIX = "mcp"


class McpServerConfig(BaseModel):
  """One external MCP server the harness may borrow tools from.

  ``transport`` is inferred when omitted: a ``url`` means http, a ``command``
  means a local stdio child process.
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  enabled: bool = True
  transport: Literal["stdio", "http"] | None = None
  # stdio (local process) —
  command: str | None = None
  args: list[str] = Field(default_factory=list)
  env: dict[str, str] = Field(default_factory=dict)
  cwd: str | None = None
  # http (streamable HTTP endpoint) —
  url: str | None = None
  headers: dict[str, str] = Field(default_factory=dict)
  # Shared —
  timeout_s: float = 60.0
  # Allowlist of the server's own tool names to expose (empty = every tool it
  # advertises). A big server is often worth borrowing three tools from.
  tools: list[str] = Field(default_factory=list)
  # Bridge the server's resources (files, tables, documents) as a list/read tool
  # pair, so a data source with no tools of its own is still readable.
  resources: bool = True
  # How this server's tools reach the model's prompt catalog:
  #   "full"  — one catalog line per tool (precise; costs prompt on every call)
  #   "index" — one line for the server, listing its tool names; the model pulls
  #             the schemas it wants with load_tool
  #   "auto"  — full for a small server, index past `catalog_threshold`
  # A real data provider can expose 90+ tools; cataloguing all of them would
  # price its entire surface into every model call, which is exactly what the
  # deferred tier exists to avoid.
  catalog: Literal["auto", "full", "index"] = "auto"
  catalog_threshold: int = 12

  def resolved_transport(self):
    """The effective transport: explicit, else inferred from url/command."""
    if self.transport:
      return self.transport
    return "http" if self.url else "stdio"

  def target(self):
    """A short human-readable description of what this server points at.

    Redacted: this string is surfaced to the UI, the ``/mcp`` response, the run's
    SSE stream, and any saved trajectory. MCP endpoints commonly carry the
    credential in the query string (``?apikey=…``), so it must not travel with
    the identity of the server.
    """
    if self.resolved_transport() == "http":
      return redact_url(self.url) if self.url else "(no url)"
    return " ".join([self.command or "(no command)", *self.args])

  def usable(self):
    """True when the config has enough to attempt a connection."""
    if self.resolved_transport() == "http":
      return bool(self.url)
    return bool(self.command)

  def expanded(self, env=None):
    """A copy with ``${VAR}`` expanded in every string field.

    Lets a config reference a credential (``"Authorization": "Bearer ${GH_TOKEN}"``)
    or a path without the secret living in the file. An undefined variable is left
    as-is rather than blanked, so a typo is visible instead of silently empty.
    """
    e = env if env is not None else os.environ

    def sub(value):
      if not isinstance(value, str):
        return value
      for key, val in e.items():
        value = value.replace(f"${{{key}}}", val)
      return value

    return self.model_copy(update={
        "command": sub(self.command),
        "args": [sub(a) for a in self.args],
        "env": {k: sub(v) for k, v in self.env.items()},
        "cwd": sub(self.cwd),
        "url": sub(self.url),
        "headers": {k: sub(v) for k, v in self.headers.items()},
    })


# Query-parameter names whose values are credentials. Matched case-insensitively
# as a substring, so `apikey`, `api_key`, `X-Api-Key` and `access_token` all hit.
_SECRET_PARAM_HINTS = (
    "key",
    "token",
    "secret",
    "password",
    "pwd",
    "auth",
    "sig",
)
_REDACTED = "***"


def redact_url(url):
  """A URL safe to display: credential query values and any userinfo masked.

  Everything else survives, because scheme/host/path is what identifies the
  server to a reader.
  """
  if not isinstance(url, str) or not url:
    return url
  try:
    parts = urlsplit(url)
  except ValueError:  # not parseable: don't risk showing it raw
    return _REDACTED
  netloc = parts.netloc
  if "@" in netloc:  # user:pass@host
    netloc = f"{_REDACTED}@{netloc.rsplit('@', 1)[1]}"
  query = parts.query
  if query:
    pairs = parse_qsl(query, keep_blank_values=True)
    query = urlencode(
        [
            (
                k,
                _REDACTED
                if any(h in k.lower() for h in _SECRET_PARAM_HINTS)
                else v,
            )
            for k, v in pairs
        ],
        # Keep the mask and an unexpanded ${VAR} legible rather than percent-encoded;
        # this string is for a human, not for dialling.
        safe="*${}",
    )
  return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _read_file(path):
  """Best-effort JSON read; returns {} on any problem (never raises)."""
  try:
    return json.loads(path.read_text())
  except Exception:  # noqa: BLE001 — config is best-effort; fall back to no servers
    return {}


def config_path(env=None):
  """The MCP config file location: ``FH_MCP_CONFIG`` env > ``configs/mcp.json``."""
  e = env if env is not None else os.environ
  override = e.get("FH_MCP_CONFIG")
  return Path(override) if override else _DEFAULT_CONFIG_PATH


def mcp_disabled(env=None):
  """True when ``FH_MCP_DISABLE`` is set to a truthy value (kill switch)."""
  e = env if env is not None else os.environ
  return (e.get("FH_MCP_DISABLE") or "").strip().lower() in {
      "1",
      "true",
      "yes",
      "on",
  }


def load_mcp_servers(path=None, env=None, *, include_disabled=False):
  """Every configured MCP server, in file order.

  ``servers`` may be a list of objects or a name → object mapping (both are
  common in MCP client configs); either shape loads. A malformed single entry is
  skipped, not fatal — one bad server never costs you the others.
  """
  if mcp_disabled(env) and not include_disabled:
    return []
  p = Path(path) if path is not None else config_path(env)
  raw: Any = _read_file(p).get("servers") if p.exists() else None
  entries: list[dict[str, Any]] = []
  if isinstance(raw, dict):
    entries = [{**spec, "name": name} for name, spec in raw.items() if isinstance(spec, dict)]
  elif isinstance(raw, list):
    entries = [spec for spec in raw if isinstance(spec, dict)]

  out: list[McpServerConfig] = []
  for spec in entries:
    try:
      cfg = McpServerConfig(**spec)
    except Exception:  # noqa: BLE001 — skip a malformed server, keep the rest
      continue
    if cfg.usable() and (include_disabled or cfg.enabled):
      out.append(cfg)
  return out
