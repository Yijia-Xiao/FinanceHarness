"""Search backends: a pluggable interface + the ddgs default.

The deep-research `search` tool talks to a backend through this small protocol,
so SearXNG / Jina / a corpus index can be added later without touching the tool.
The core ships `DdgsBackend` (the `ddgs` library, zero infra) as the default and
an opt-in `YouComBackend`; other backends (SearXNG, Jina, a corpus index) plug in
through the same protocol. `search.backend` in `runtime.json` (or
`FH_SEARCH_BACKEND`) picks one by name.
"""

from __future__ import annotations

import asyncio
import os
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

    async def search(self, query, max_results): ...


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


_YOU_KEYLESS_URL = "https://api.you.com/v1/agents/search"
_YOU_KEYED_URL = "https://ydc-index.io/v1/search"
_YOU_KEYLESS_MAX_RESULTS = 50
_YOU_USER_AGENT = "youdotcom-integration/Yijia-Xiao-FinanceHarness"


class YouComBackend:
    """You.com Search API, opt-in through `search.backend = "youcom"`.

    Without `YDC_API_KEY` it uses the keyless endpoint (no account, about 100
    searches a day, up to 50 results per query). With the key set it uses the
    keyed API. Only the query and a `User-Agent` header are sent to You.com.
    """

    name = "youcom"

    def __init__(self, api_key=None, timeout=15.0):
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query, max_results):
        import httpx

        api_key = self._api_key if self._api_key is not None else os.environ.get("YDC_API_KEY", "")
        headers = {"User-Agent": _YOU_USER_AGENT}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if api_key:
                resp = await client.post(
                    _YOU_KEYED_URL,
                    json={"query": query, "count": max_results},
                    headers={**headers, "X-API-Key": api_key},
                )
            else:
                resp = await client.get(
                    _YOU_KEYLESS_URL,
                    params={"query": query, "count": min(max_results, _YOU_KEYLESS_MAX_RESULTS)},
                    headers=headers,
                )
        resp.raise_for_status()
        rows = (resp.json().get("results") or {}).get("web") or []
        out: list[SearchResult] = []
        for r in rows:
            url = r.get("url") or ""
            if not url:
                continue
            snippet = " ".join(r.get("snippets") or []) or r.get("description") or ""
            out.append(
                SearchResult(
                    title=(r.get("title") or "").strip(),
                    url=url,
                    snippet=snippet.strip(),
                )
            )
        return out[:max_results]


_BACKENDS = {"ddgs": DdgsBackend, "youcom": YouComBackend}


def default_backend(config=None):
    """The backend named by `search.backend` in the runtime config (ddgs by default)."""
    from financeharness.runtime.config import load_runtime_config

    cfg = config or load_runtime_config()
    return _BACKENDS.get(cfg.search.backend, DdgsBackend)()
