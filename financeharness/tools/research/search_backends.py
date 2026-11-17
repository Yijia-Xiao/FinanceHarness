"""Search backends: a pluggable interface + the ddgs default.

The deep-research `search` tool talks to a backend through this small protocol,
so SearXNG / Jina / a corpus index can be added later without touching the tool.
The core ships `DdgsBackend` (the `ddgs` library, zero infra); other backends
(SearXNG, Jina, a corpus index) plug in through the same protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
  """One search hit returned by a backend."""

  title: str
  url: str
  snippet: str


class SearchBackend(Protocol):
  """Minimal async protocol implemented by search providers."""

  name: str

  async def search(self, query, max_results):
    ...


class DdgsBackend:
  """DuckDuckGo via the `ddgs` library.

  Synchronous under the hood, so the blocking call runs in a worker thread.
  """

  name = "ddgs"

  async def search(self, query, max_results):
    return await asyncio.to_thread(self._search_sync, query, max_results)

  @classmethod
  def _search_sync(cls, query, max_results):
    """Runs synchronous search using the DDGS library."""
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results)
    out: list[SearchResult] = []
    for r in rows:
      url = r.get("href") or r.get("url") or ""
      if not url:
        continue
      out.append(
          SearchResult(
              title=(r.get("title") or "").strip(),
              url=url,
              snippet=(r.get("body") or "").strip(),
          )
      )
    return out
