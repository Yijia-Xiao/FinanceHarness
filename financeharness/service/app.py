"""FastAPI app — research/clarify/compact endpoints, sessions, health, MCP.

`POST /research` serves both the sync JSON path and the SSE stream
(stream=true).
`run_research` is module-level so a caller can substitute a fake (a headless
client needs no live LLM).

`GET /mcp` reports the configured external MCP servers and what they expose, so
a client can show which data sources a run can actually reach.

When the React frontend has been built (``web/dist``), it is served at ``/`` —
one process then serves the API and the UI. Absent a build, the API runs alone.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from financeharness.clarify import scope_question, scope_question_stream
from financeharness.mcp import McpHub, config_path, load_mcp_servers, mcp_disabled
from financeharness.providers import (
    available_backbones,
    client_for,
    get_profile,
    load_profiles,
)
from financeharness.providers.profiles import default_profile_name
from financeharness.research import run_research
from financeharness.runtime.config import load_runtime_config
from financeharness.runtime.context_budget import estimate_tokens
from financeharness.runtime.modes import resolve_mode
from financeharness.runtime.summarize import summarize_history
from financeharness.service.events import SSE_VERSION, sse_frame
from financeharness.service.sessions import SessionStore, default_sessions_dir

_SENTINEL = object()
_SESSIONS = SessionStore(default_sessions_dir())

# SSE keepalive: tools (visit/search) and non-streaming model calls can run for
# tens of seconds with no events; without a periodic byte the client's stream can
# idle-timeout. A comment frame (ignored by parsers) keeps the connection alive.
_HEARTBEAT_S = 10.0
_PING_FRAME = ": ping\n\n"


async def _drain(queue):
  """Yield SSE frames from the queue, emitting a heartbeat during silent gaps,

  until the sentinel arrives.
  """
  while True:
    try:
      kind, data = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
    except TimeoutError:
      yield _PING_FRAME
      continue
    if kind is _SENTINEL:
      break
    yield sse_frame(kind, data)


API_VERSION = "0.1"

app = FastAPI(title="FinanceHarness", version=API_VERSION)


class ResearchRequest(BaseModel):
  """Request body for one research turn, sync or streamed."""

  question: str = Field(Ellipsis, min_length=1, description="The research question.")
  equity: bool = Field(
      False, description="(Legacy) enable equity tools; prefer `mode`."
  )
  mode: str | None = Field(
      None,
      description=(
          "Execution mode loadout: auto | research | analytical (default research)."
      ),
  )
  profile: str | None = Field(
      None, description="Backbone profile override (default: the configured default)."
  )
  stream: bool = Field(
      False, description="Stream progress as Server-Sent Events."
  )
  session_id: str | None = Field(
      None,
      description=(
          "Thread this turn into a multi-turn session (omit for one-shot)."
      ),
  )
  clarifications: list[dict[str, Any]] | None = Field(
      None,
      description=(
          "Answered clarifying questions [{question, answer}] to weave into"
          " scope."
      ),
  )


class ClarifyRequest(BaseModel):
  """Request body for the pre-research scoping pass."""

  question: str = Field(Ellipsis, min_length=1, description="The question to scope.")
  profile: str | None = Field(None, description="Backbone profile override.")
  stream: bool = Field(
      False, description="Stream scoping progress as Server-Sent Events."
  )


class CompactRequest(BaseModel):
  """Request body for compacting a durable session history."""

  session_id: str = Field(
      Ellipsis, description="The session to compact (summarize older history)."
  )
  instruction: str | None = Field(
      None,
      description=(
          "Optional steer for the summary (e.g. 'keep every ticker figure')."
      ),
  )
  profile: str | None = Field(None, description="Backbone profile override.")


def _backbone(name):
  """Resolve a backbone profile override by name, or ``None`` when unset.

  Rejects an unknown name or a reader-role profile with HTTP 400 (mirrors the
  CLI's ``--profile`` validation) instead of ``get_profile()``'s silent
  fallback to the default.
  """
  if not name:
    return None
  profiles = load_profiles()
  p = profiles.get(name)
  if p is None or p.role != "backbone":
    backbones = sorted(n for n, q in profiles.items() if q.role == "backbone")
    raise HTTPException(
        status_code=400,
        detail=(
            f"unknown backbone profile '{name}'; available: "
            + ", ".join(backbones)
        ),
    )
  return p


async def _probe(profile):
  """True if the profile's resolved endpoint answers an authenticated ``/models``.

  Sends the API key (a cloud ``/models`` 401s unauthenticated → false offline);
  a keyless local vLLM (key ``EMPTY``) is probed without auth.
  """
  base = str(client_for(profile).base_url).rstrip("/")
  key = profile.resolve_api_key()
  headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}
  try:
    async with httpx.AsyncClient(timeout=3.0) as client:
      r = await client.get(base + "/models", headers=headers)
      return r.status_code == 200
  except Exception:  # noqa: BLE001 — unreachable endpoint → not ready
    return False


@app.get("/health")
async def health(profile = None):
  """Readiness of a backbone + its paired reader.

  ``profile`` selects which backbone to report (default: the configured
  default), so a client can reflect the backbone it is actually using rather
  than always the default.
  """
  backbone = _backbone(profile) or get_profile()
  reader = get_profile(backbone.reader_profile or "qwen-reader")
  dr_ready = await _probe(backbone)
  reader_ready = await _probe(reader)
  return {
      "status": "ok" if (dr_ready and reader_ready) else "degraded",
      "version": API_VERSION,
      "backbone": {
          "profile": backbone.name,
          "model": backbone.model,
          "ready": dr_ready,
      },
      "reader": {
          "profile": reader.name,
          "model": reader.model,
          "ready": reader_ready,
      },
  }


async def _event_stream(
    req,
    profile,
    history,
):
  """Bridge the agent's on_event callback into an SSE frame stream.

  The run executes as a task; its (sync) on_event pushes events onto a queue
  that this generator drains. The agent's internal `done` (termination only) is
  dropped — the protocol's terminal `done` carries the full trajectory.
  """
  queue: asyncio.Queue = asyncio.Queue()

  def on_event(kind, data):
    if kind == "done":
      return  # superseded by the protocol's terminal done (with trajectory)
    queue.put_nowait((kind, data))

  async def runner():
    try:
      traj = await run_research(
          req.question,
          profile=profile,
          equity=req.equity,
          mode=req.mode,
          history=history,
          clarifications=req.clarifications,
          on_event=on_event,
          stream_tokens=True,
      )
      _SESSIONS.commit(req.session_id, traj)
      queue.put_nowait(
          ("done", {"trajectory": traj, "session_id": req.session_id})
      )
    except Exception as exc:  # noqa: BLE001 — surface as an SSE error + empty done
      queue.put_nowait(("error", {"error": f"{type(exc).__name__}: {exc}"}))
      queue.put_nowait(
          ("done", {"trajectory": None, "session_id": req.session_id})
      )
    finally:
      queue.put_nowait((_SENTINEL, None))

  task = asyncio.create_task(runner())
  yield sse_frame(
      "run_start",
      {
          "version": SSE_VERSION,
          "question": req.question,
          "equity": req.equity,
          "mode": resolve_mode(req.mode, equity=req.equity),
      },
  )
  try:
    async for frame in _drain(queue):
      yield frame
  finally:
    if not task.done():
      task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _clarify_stream(
    req, profile
):
  """Bridge the scoping pass's on_event callback into an SSE frame stream — the

  context search step + hits stream first, then a terminal `clarify_result`.
  """
  queue: asyncio.Queue = asyncio.Queue()

  def on_event(kind, data):
    if kind == "clarify_result":
      return  # superseded by the terminal frame (below)
    queue.put_nowait((kind, data))

  async def runner():
    try:
      result = await scope_question_stream(
          req.question, profile=profile, on_event=on_event
      )
      queue.put_nowait(("clarify_result", {"result": result.model_dump()}))
    except Exception as exc:  # noqa: BLE001 — fail-open: emit a sufficient result
      queue.put_nowait(("error", {"error": f"{type(exc).__name__}: {exc}"}))
      queue.put_nowait(("clarify_result", {"result": {"sufficient": True}}))
    finally:
      queue.put_nowait((_SENTINEL, None))

  task = asyncio.create_task(runner())
  try:
    async for frame in _drain(queue):
      yield frame
  finally:
    if not task.done():
      task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@app.post("/clarify")
async def clarify(req: ClarifyRequest):
  """Scope a question: decide if it's specific enough, else return clarifying

  questions. Sync JSON by default; SSE when stream=true. Fail-open. Scoping is a
  single lightweight backbone call (fail-open if the backbone isn't reachable).
  """
  profile = _backbone(req.profile)
  if req.stream:
    return StreamingResponse(
        _clarify_stream(req, profile), media_type="text/event-stream"
    )
  result = await scope_question(req.question, profile=profile)
  return result.model_dump()


@app.post("/compact")
async def compact(req: CompactRequest):
  """Compact a multi-turn session: LLM-summarize the older history (keeping the

  recent turn verbatim) and swap it in, freeing context. Optional `instruction`
  steers what to preserve. Fail-open (returns compacted=false on a no-op/error).
  """
  history = _SESSIONS.history(req.session_id)
  if not history:
    return {"compacted": False, "reason": "unknown or empty session"}
  profile = _backbone(req.profile) or get_profile()
  new_history, stats = await summarize_history(
      history, instruction=req.instruction, profile=profile
  )
  if stats.get("compacted"):
    _SESSIONS.replace(req.session_id, new_history)
  return stats


@app.get("/sessions")
async def sessions():
  """List saved sessions (most-recently-updated first) for a client to browse.

  Summaries only — id, title, timestamps, turn count — no message bodies.
  """
  return {"sessions": _SESSIONS.list()}


@app.get("/sessions/{session_id}")
async def session_detail(session_id):
  """A session's stored conversation (system stripped) — lets a client rehydrate

  the transcript when resuming. Empty messages for an unknown session.
  """
  return {"id": session_id, "messages": _SESSIONS.history(session_id) or []}


@app.get("/models")
async def models():
  """Switchable backbone profiles + credential availability, for a client's

  model picker. The default is the configured default backbone (gemini); other
  configured profiles appear as available once their credentials are set.
  """
  return {"models": available_backbones(), "default": default_profile_name()}


@app.get("/mcp")
async def mcp(probe: bool = True):
  """The external MCP servers this harness can borrow tools from.

  ``probe=true`` (the default) connects to each configured server and reports
  what it actually exposes — the honest answer to "can this run reach my data?".
  ``probe=false`` returns the configuration alone (no child processes started, no
  endpoints touched), which is the cheap call for a page load.
  """
  configured = load_mcp_servers(include_disabled=True)
  summary = {
      "enabled": not mcp_disabled(),
      "config_path": str(config_path()),
      "configured": [
          {
              "name": c.name,
              "transport": c.resolved_transport(),
              "target": c.target(),
              "enabled": c.enabled,
              "tool_allowlist": c.tools,
          }
          for c in configured
      ],
  }
  active = [c for c in configured if c.enabled] if not mcp_disabled() else []
  if not probe or not active:
    summary["servers"] = []
    summary["probed"] = False
    return summary
  hub = await McpHub.connect(active)
  try:
    summary["servers"] = hub.status()
    summary["tool_count"] = len(hub.specs())
  finally:
    await hub.aclose()
  summary["probed"] = True
  return summary


@app.get("/status")
async def status(
    session_id = None, profile = None
):
  """Context usage for a session (token count vs window) for a client's meter.

  ``used_pct`` is of the configured context window; the model label reflects the
  active backbone (``profile``), defaulting to the configured one.
  """
  cfg = load_runtime_config()
  backbone = _backbone(profile) or get_profile()
  window = cfg.recovery.context_window
  used = estimate_tokens(_SESSIONS.history(session_id))
  return {
      "model": backbone.model,
      "context_window": window,
      "tokens_used": used,
      "used_pct": round(100 * used / window, 1) if window else 0.0,
      "turns": next(
          (s["turns"] for s in _SESSIONS.list() if s["id"] == session_id), 0
      ),
  }


@app.post("/research")
async def research(req: ResearchRequest):
  """Run one research trajectory. Sync JSON by default; SSE when stream=true.

  Pass ``session_id`` to thread a session; ``clarifications`` to weave scope
  answers.
  """
  profile = _backbone(req.profile)
  history = _SESSIONS.history(req.session_id)
  if req.stream:
    return StreamingResponse(
        _event_stream(req, profile, history), media_type="text/event-stream"
    )
  traj = await run_research(
      req.question,
      profile=profile,
      equity=req.equity,
      mode=req.mode,
      history=history,
      clarifications=req.clarifications,
  )
  _SESSIONS.commit(req.session_id, traj)
  if req.session_id:
    traj = {**traj, "session_id": req.session_id}
  return traj


# --- the web UI -------------------------------------------------------------
# Mounted last so every API route above wins; a client-side-routed SPA needs its
# index served for unknown paths, hence the explicit fallback. Skipped entirely
# when the frontend hasn't been built — `fh serve` is useful API-only.
_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

if _WEB_DIST.is_dir():
  app.mount(
      "/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets"
  )

  @app.get("/")
  async def index():
    return FileResponse(_WEB_DIST / "index.html")

  @app.get("/{path:path}", include_in_schema=False)
  async def spa_fallback(path):
    """Serve a real file when it exists, else the SPA shell."""
    candidate = (_WEB_DIST / path).resolve()
    if candidate.is_file() and _WEB_DIST in candidate.parents:
      return FileResponse(candidate)
    return FileResponse(_WEB_DIST / "index.html")
