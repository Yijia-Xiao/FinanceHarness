"""Citation post-processing — pure end-of-run validation + bibliography.

Applied at the single exit (`Agent._build_result` via the `finalize` hook):
strip any model-written References section and leaked markers, drop orphan
`[N]` markers (numbers past the cache size), and append the canonical
`## References` block when sources exist. When the cache is empty, no block is
appended — the body is the complete deliverable.

Pure functions over the citation list; no I/O.
"""

from __future__ import annotations

import re

# A model-written reference section: a heading, a bold label, or a bare
# "References" line followed by a list entry. A bare line followed by prose is
# part of the report.
_REF_SECTION_RE = re.compile(
    r"\n+(?:##+\s*references\b"
    r"|\*\*\s*references\s*\*\*"
    r"|references:?[ \t]*\n(?=\s*(?:\[\d|https?://|\d+[.)]\s|[-*+]\s)))",
    flags=re.IGNORECASE,
)
# Strip leaked "[End of output … appends … references …]" template markers.
_EOM_LEAK_RE = re.compile(
    r"\n*\[?\s*End of (?:your )?(?:output|report)\."
    r"[^\]\n]*?(?:appends?)[^\]\n]*?references[^\]\n]*?\]?\s*",
    flags=re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"(?P<lead>\s*)\[(?P<digits>\d+(?:\s*,\s*\d+)*)\]")
# Fenced blocks and inline code spans: bracketed numbers inside them are code
# (``arr[0]``), not citation markers.
_CODE_RE = re.compile(r"```.*?```|`[^`\n]+`", flags=re.DOTALL)


def _markers(body):
  """Citation marker matches in ``body`` outside code."""
  code = [m.span() for m in _CODE_RE.finditer(body)]
  return [
      m
      for m in _BRACKET_RE.finditer(body)
      if not any(start <= m.start("digits") < end for start, end in code)
  ]


def format_references_block(citations):
  """Render the bibliography: ``[N] title`` + ``    url`` per source."""
  lines: list[str] = []
  for c in citations:
    title = getattr(c, "title", None) or getattr(c, "url", "") or "(untitled)"
    url = getattr(c, "url", "")
    lines.append(f"[{c.index}] {title}")
    if url:
      lines.append(f"    {url}")
  return "\n".join(lines)


def validate_and_append_references(
    prediction, citations
):
  """Validate inline markers, strip orphans + model-written refs, and append

  the canonical ``## References`` block. Returns ``(text, stats)``.
  """
  m = _REF_SECTION_RE.search(prediction)
  body = (prediction[: m.start()] if m else prediction).rstrip()
  body = _EOM_LEAK_RE.sub("\n", body).rstrip()

  n_refs = len(citations)
  cited: set[int] = set()
  for match in _markers(body):
    for n in match.group("digits").split(","):
      with_int = n.strip()
      if with_int.isdigit():
        cited.add(int(with_int))

  detected_orphans = sorted(n for n in cited if n < 1 or n > n_refs)
  valid_cited = sorted(n for n in cited if 1 <= n <= n_refs)

  if detected_orphans:
    parts: list[str] = []
    pos = 0
    for match in _markers(body):
      nums = [
          d.strip()
          for d in match.group("digits").split(",")
          if 1 <= int(d.strip()) <= n_refs
      ]
      parts.append(body[pos : match.start()])
      if nums:
        parts.append(f"{match.group('lead')}[{', '.join(nums)}]")
      pos = match.end()
    parts.append(body[pos:])
    body = "".join(parts)

  if not citations:
    return body + "\n", {
        "n_refs": 0,
        "n_inline_citations": len(valid_cited),
        "cited_indices": valid_cited,
        "n_orphans_stripped": len(detected_orphans),
        "unused_indices": [],
    }

  unused = sorted(n for n in range(1, n_refs + 1) if n not in valid_cited)
  out = (
      body + "\n\n## References\n\n" + format_references_block(citations) + "\n"
  )
  return out, {
      "n_refs": n_refs,
      "n_inline_citations": len(valid_cited),
      "cited_indices": valid_cited,
      "n_orphans_stripped": len(detected_orphans),
      "unused_indices": unused,
  }
