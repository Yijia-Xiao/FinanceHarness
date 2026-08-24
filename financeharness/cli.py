"""`financeharness` (alias `fh`) — the command-line interface.

Headless one-shot research (progress to stderr, report to stdout, pipeable):

    financeharness "NVDA DCF value?"          # run the question, print the
    report
    financeharness -p "..." --save run.json --profile gemini
    echo "..." | financeharness -p            # question piped via stdin
    financeharness --list                     # show profiles + skills + MCP
    financeharness serve                      # the HTTP+SSE backend (and web UI)
    financeharness mcp                        # serve the harness over MCP (stdio)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from financeharness.mcp import load_mcp_servers, mcp_disabled
from financeharness.providers import get_profile, load_profiles
from financeharness.research import run_research, save_trajectory
from financeharness.tools.research import default_skill_registry


def _err(msg: str) -> None:
  print(msg, file=sys.stderr, flush=True)


def _make_progress(quiet: bool):
  def on_event(kind: str, data: dict[str, Any]) -> None:
    if quiet:
      return
    if kind == "round_start":
      _err(f"  ── round {data['round']}")
    elif kind == "tool_call":
      _err(f"     → {data['name']}")
    elif kind == "tool_result":
      _err(f"       {'ok' if data['ok'] else 'FAIL'}")
    elif kind == "error":
      _err(f"  ! {data.get('error', 'error')}")

  return on_event


async def _research(args: argparse.Namespace) -> int:
  profile = get_profile(args.profile)
  reader = get_profile(args.reader) if args.reader else None
  label = args.mode or ("analytical (--equity)" if args.equity else "research")
  _err(f"[fh] {profile.model} · {label} · researching…")
  traj = await run_research(
      args.question,
      profile=profile,
      reader_profile=reader,
      mode=args.mode,  # primary; falls back to the legacy `equity` flag when unset
      equity=args.equity,
      on_event=_make_progress(args.quiet),
  )
  _err(
      f"[fh] {traj['termination']} · {traj['rounds']} rounds · "
      f"{len(traj['citations'])} sources · {traj['elapsed_s']}s"
  )
  if args.save:
    _err(f"[fh] saved {save_trajectory(traj, args.save)}")
  print("\n" + (traj["prediction"] or "(no answer produced)"))
  return 0 if traj["termination"] == "answer" else 1


def _serve(argv: list[str]) -> int:
  """`fh serve` — run the HTTP+SSE service.

  Use --reload in dev so code changes are picked up (plain uvicorn does NOT
  auto-reload — a restart is otherwise required).
  """
  ap = argparse.ArgumentParser(
      prog="financeharness serve", description="Run the FinanceHarness service"
  )
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument(
      "--reload", action="store_true", help="auto-reload on code changes (dev)"
  )
  args = ap.parse_args(argv)
  import uvicorn

  _err(
      "[fh] serving on"
      f" http://{args.host}:{args.port}{' (reload)' if args.reload else ''}"
  )
  uvicorn.run(
      "financeharness.service.app:app",
      host=args.host,
      port=args.port,
      reload=args.reload,
  )
  return 0


def _mcp(argv: list[str]) -> int:
  """`fh mcp` — expose the harness itself over the Model Context Protocol.

  stdio by default, which is what an MCP host (Claude Desktop, an IDE) spawns.
  In stdio mode stdout *is* the protocol channel, so every message this function
  prints goes to stderr.
  """
  ap = argparse.ArgumentParser(
      prog="financeharness mcp",
      description="Serve FinanceHarness tools, skills and deep_research over MCP",
  )
  ap.add_argument(
      "--http",
      action="store_true",
      help="serve over streamable HTTP instead of stdio (for a remote client)",
  )
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--port", type=int, default=8765)
  ap.add_argument(
      "--path", default="/mcp", help="HTTP path for the MCP endpoint (--http)"
  )
  ap.add_argument(
      "--list",
      action="store_true",
      help="print the tools/skills that would be exposed, then exit",
  )
  ap.add_argument(
      "--profile", default=None, help="backbone for deep_research (default: configured)"
  )
  args = ap.parse_args(argv)

  from financeharness.mcp.server import build_server, describe_surface

  if args.list:
    surface = describe_surface()
    print(f"FinanceHarness MCP server — backbone {surface['backbone']}")
    print(f"\nTools ({len(surface['tools'])}):")
    for tool in surface["tools"]:
      print(f"  {tool['name']}")
    print("\nSkills (exposed as MCP prompts + resources):")
    for name in surface["skills"]:
      print(f"  {name}")
    if surface.get("skipped"):
      print(f"\nNot exposed (unbridgeable schema): {', '.join(surface['skipped'])}")
    return 0

  server, session = build_server(
      backbone=get_profile(args.profile) if args.profile else None
  )
  if args.http:
    _err(f"[fh] MCP over HTTP at http://{args.host}:{args.port}{args.path}")
    asyncio.run(
        server.run_streamable_http_async(
            host=args.host, port=args.port, streamable_http_path=args.path
        )
    )
    return 0
  _err(f"[fh] MCP on stdio · backbone {session.backbone.name} · ready")
  asyncio.run(server.run_stdio_async())
  return 0


def _mcp_client_lines() -> list[str]:
  """The configured *outbound* MCP servers, for `--list` (config only, no dial)."""
  if mcp_disabled():
    return ["  (disabled by FH_MCP_DISABLE)"]
  servers = load_mcp_servers(include_disabled=True)
  if not servers:
    return ["  (none configured — see configs/mcp.json)"]
  return [
      f"  {'  ' if s.enabled else '× '}{s.name} [{s.resolved_transport()}]: {s.target()}"
      for s in servers
  ]


def _list() -> int:
  # Backbones only — the selectable orchestrator profiles (readers are paired
  # internally and are not a `--profile` choice), matching --profile validation.
  print("Backbones (default marked *):")
  profiles = load_profiles()
  default = get_profile().name
  for name in sorted(profiles):
    p = profiles[name]
    if p.role != "backbone":
      continue
    mark = " *" if name == default else "  "
    print(f"{mark} {name}: {p.model}")
  print("\nSkills (the model loads these on demand):")
  for s in default_skill_registry().all():
    print(f"   {s.name}: {s.description}")
  print("\nMCP servers (external tools this harness can borrow):")
  for line in _mcp_client_lines():
    print(line)
  return 0


def main() -> None:
  """CLI entrypoint.

  Headless one-shot research; ``serve`` for the backend alone.
  """

  # `serve` and `mcp` are sub-commands; everything else is the research parser.
  if sys.argv[1:2] == ["serve"]:
    raise SystemExit(_serve(sys.argv[2:]))
  if sys.argv[1:2] == ["mcp"]:
    raise SystemExit(_mcp(sys.argv[2:]))

  ap = argparse.ArgumentParser(
      prog="financeharness",
      description="FinanceHarness — finance deep-research agent",
  )
  ap.add_argument(
      "question", nargs="?", help="the research question (or pipe via stdin)"
  )
  ap.add_argument(
      "-p",
      "--print",
      dest="oneshot",
      action="store_true",
      help=(
          "one-shot (default): run the question, print the report to stdout,"
          " and exit"
      ),
  )
  ap.add_argument(
      "--mode",
      choices=["auto", "research", "analytical"],
      default=None,
      help=(
          "execution mode (default: research). auto = web + tools; analytical ="
          " numbers-first."
      ),
  )
  ap.add_argument(
      "--equity",
      action="store_true",
      help="(legacy alias for --mode analytical)",
  )
  ap.add_argument(
      "--profile",
      default=None,
      help="orchestrator profile (default: the configured default)",
  )
  ap.add_argument(
      "--reader",
      default=None,
      help="page-reader profile (default: paired with the backbone)",
  )
  ap.add_argument(
      "--save", default=None, help="save the trajectory JSON to this path"
  )
  ap.add_argument(
      "--quiet", action="store_true", help="suppress progress (report only)"
  )
  ap.add_argument(
      "--list", action="store_true", help="list profiles + skills and exit"
  )
  args = ap.parse_args()

  if args.list:
    raise SystemExit(_list())

  # Validate explicit profile selections up front: an unknown name would
  # otherwise fall back to the default silently (a typo like `--profile gemni`
  # would run on the wrong backbone). --profile must name a backbone (not a
  # reader); --reader may name any profile. Fail loudly with the valid names.
  known = load_profiles()
  backbones = sorted(n for n, p in known.items() if p.role == "backbone")
  if args.profile and args.profile not in backbones:
    ap.error(
        f"unknown --profile '{args.profile}'; available: {', '.join(backbones)}"
    )
  if args.reader and args.reader not in known:
    ap.error(
        f"unknown --reader '{args.reader}'; available: {', '.join(sorted(known))}"
    )

  # Headless one-shot: question from the positional arg or piped stdin.
  if not args.question and not sys.stdin.isatty():
    args.question = sys.stdin.read().strip()
  if not args.question:
    ap.error(
        "a question is required (positional, piped via stdin, or use --list)"
    )
  raise SystemExit(asyncio.run(_research(args)))


if __name__ == "__main__":
  main()
