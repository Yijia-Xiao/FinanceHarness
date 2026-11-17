"""The `compose_citations` core tool: hand the model its numbered bibliography.

Returns the visit-order, URL-deduped citation list so the model can place inline
`[N]` markers in its report. The harness appends the matching `## References`
block at finalize regardless, so this tool is the model's cue for *where* to
cite, not the bibliography's source of truth.
"""

from __future__ import annotations

from pydantic import BaseModel

from financeharness.runtime.citations import format_references_block
from financeharness.runtime.tool_registry import ToolResponse, ToolSpec

_DESCRIPTION = (
    "Return the numbered list of sources you have read with `visit` (visit"
    " order, deduped). Call it just before writing the final report, then use"
    " the numbers as inline [N] markers; the harness appends the matching"
    " References section automatically."
)

_WRITE_INSTRUCTION = (
    "\n\nNow write the final report, using the numbers above as inline [N]"
    " markers next to each claim drawn from a source. The harness appends the"
    " References section from these sources, so your body with inline markers"
    " is the complete deliverable."
)

_EMPTY_INSTRUCTION = (
    "No sources have been read yet. Research further with search/visit, or (if"
    " answering from prior knowledge) write the report as plain prose; the"
    " body alone is the complete deliverable (the harness adds no bibliography"
    " here)."
)


class ComposeCitationsRequest(BaseModel):
  """Input for composing the current run's citation list."""


def build_citations_spec(cache):
  """Build the per-run `compose_citations` tool bound to the cache."""

  async def handler(_req):
    citations = cache.citations
    if not citations:
      return ToolResponse(markdown=_EMPTY_INSTRUCTION, structured={"count": 0})
    block = format_references_block(citations)
    return ToolResponse(
        markdown=block + _WRITE_INSTRUCTION,
        structured={
            "count": len(citations),
            "citations": [c.url for c in citations],
        },
    )

  return ToolSpec(
      name="compose_citations",
      display_name="compose_citations",
      tier="core",
      description=_DESCRIPTION,
      request_schema=ComposeCitationsRequest,
      handler=handler,
      tags=("research", "citations"),
  )
