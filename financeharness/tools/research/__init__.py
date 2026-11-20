"""Deep-research tools: search (discovery), visit (retrieval), compose_citations.

Built per run via factories that share a :class:`FetchCache`.
"""

from financeharness.tools.research.assembly import (
    build_equity_research_registry,
    build_research_registry,
    citation_finalizer,
    default_skill_registry,
)
from financeharness.tools.research.cache import Citation, FetchCache
from financeharness.tools.research.citations import build_citations_spec
from financeharness.tools.research.search import build_search_spec
from financeharness.tools.research.search_backends import (
    DdgsBackend,
    SearchBackend,
    SearchResult,
    YouComBackend,
    default_backend,
)
from financeharness.tools.research.visit import build_visit_spec
from financeharness.tools.research.visit_fetch import FetchedPage

__all__ = [
    "Citation",
    "DdgsBackend",
    "FetchCache",
    "FetchedPage",
    "SearchBackend",
    "SearchResult",
    "YouComBackend",
    "build_citations_spec",
    "build_equity_research_registry",
    "build_research_registry",
    "build_search_spec",
    "build_visit_spec",
    "citation_finalizer",
    "default_backend",
    "default_skill_registry",
]
