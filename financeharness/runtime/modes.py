"""Execution modes: each a system-prompt variant over a CONSISTENT full tool surface.

Mode is a request field selecting the prompt; the tool registry is the same for
every mode (web tools visible + equity/valuation tools deferred-but-callable +
load_tool). Keeping the toolset constant means switching mode mid-session never
strands the model with history that references a now-missing tool.

  auto        full toolkit; the agent decides how to research.
  research    web-first deep research; research prompt.
  analytical  numbers-first: lead with data/compute tools (compute, not in
              head).

The prompt sets the emphasis; the deferred tier keeps the visible schema lean
per
mode without removing tools.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODE = "auto"


@dataclass(frozen=True)
class ModeSpec:
    """Execution-mode metadata used by the service, CLI, and prompt builder."""

    name: str
    prompt_variant: str  # key into prompts.PROMPT_VARIANTS
    description: str


MODES: dict[str, ModeSpec] = {
    "auto": ModeSpec("auto", "auto", description="Full toolkit; the agent decides."),
    "research": ModeSpec(
        "research",
        "research",
        description="Web-first deep research (search, visit, cite).",
    ),
    "analytical": ModeSpec(
        "analytical",
        "analytical",
        description="Numbers-first: data + valuation tools.",
    ),
}


def get_mode(name):
    """Resolve a mode name to its spec; unknown / missing → the default mode."""
    return MODES.get((name or "").strip().lower(), MODES[DEFAULT_MODE])


def resolve_mode(name):
    """Effective mode name from an explicit `name`, else web-first research.

    The single source the ``run_start`` frame and the agent's prompt variant both
    use, so the advertised mode always matches the one that runs. An explicit name
    is normalized (unknown → the default mode).
    """
    if name and name.strip():
        return get_mode(name).name
    return "research"
