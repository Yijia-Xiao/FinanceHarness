"""FetchCache: per-run shared state for the research tools.

`search` records discovered titles; `visit` records fetched page content and
appends citations; `compose_citations` reads the citation index. One cache per
agent loop, threaded into the tool factories, so the trio coordinate without
global state.

The citation index is the single source of the bibliography: a URL is cited
only when `visit` successfully retrieved an accessible page, in visit order,
deduped by URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
  """One visited source in the run-local bibliography."""

  index: int  # 1-based, the [N] marker
  url: str
  title: str


@dataclass
class FetchCache:
  """Run-local page-content, title, and citation cache shared by research tools."""

  _content: dict[str, str] = field(default_factory=dict)
  _titles: dict[str, str] = field(default_factory=dict)
  _citations: list[Citation] = field(default_factory=list)
  _cited_index: dict[str, int] = field(default_factory=dict)

  # titles (set by search + visit) -------------------------------------- #
  def set_title(self, url, title):
    if title:
      self._titles[url] = title

  def get_title(self, url):
    return self._titles.get(url)

  # content (set by visit; read on cache hit) --------------------------- #
  def set_content(self, url, text):
    self._content[url] = text

  def get_content(self, url):
    return self._content.get(url)

  def has_content(self, url):
    return url in self._content

  # citations (appended by visit; read by compose_citations) ------------ #
  def add_citation(self, url, title = None):
    """Add `url` to the bibliography (dedup by URL, visit order).

    Returns its 1-based marker index.
    """
    if url in self._cited_index:
      return self._cited_index[url]
    index = len(self._citations) + 1
    self._citations.append(
        Citation(
            index=index, url=url, title=title or self._titles.get(url) or url
        )
    )
    self._cited_index[url] = index
    return index

  @property
  def citations(self):
    return list(self._citations)
