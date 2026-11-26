"""Search reports fallback leads to the model when readability checks fail."""

import pytest

from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.research.cache import FetchCache
from financeharness.tools.research.search import _MAX_VALIDATE, build_search_spec
from financeharness.tools.research.search_backends import SearchResult
from financeharness.tools.research.visit_fetch import FetchedPage
from tests.support import run


class StaticSearch:
    name = "fixture"

    def __init__(self, count):
        self.hits = [
            SearchResult(f"Page {i}", f"https://example.org/{i}", "Search snippet")
            for i in range(count)
        ]

    async def search(self, query, max_results):
        return self.hits[:max_results]


@pytest.mark.parametrize(
    ("readable", "returned", "notice"),
    [
        ({0, 1}, [0, 1], None),
        ({1}, [1], None),
        (set(), [0, 1], "Readability checks passed: 0 of 2."),
        (None, [0, 1], None),
    ],
)
def test_search_readability_in_model_output(readable, returned, notice):
    backend = StaticSearch(2)

    async def validate(url):
        long_enough = int(url.rsplit("/", 1)[1]) in readable
        return FetchedPage(url=url, text="text " * 50 if long_enough else "Short", ok=True)

    registry = ToolRegistry()
    cache = FetchCache()
    registry.register(
        build_search_spec(cache, backend, validator=validate if readable is not None else None)
    )
    result = run(dispatch("search", {"queries": ["example"]}, registry=registry))
    assert result.ok
    assert [r["url"] for r in result.structured["results"]] == [
        backend.hits[i].url for i in returned
    ]
    if notice:
        assert notice in result.for_model()
        assert "unverified leads" in result.for_model()
    else:
        assert "unverified leads" not in result.for_model()
        assert "Readability checks passed" not in result.for_model()
    if readable is not None:
        assert result.meta["validated"] == len(readable)
        assert result.meta["dropped"] == len(backend.hits) - len(readable)
    assert [i for i, hit in enumerate(backend.hits) if cache.has_content(hit.url)] == sorted(
        readable or ()
    )
    assert "`visit`" in result.for_model()


def test_warning_counts_checked_links_when_the_batch_exceeds_the_validation_limit():
    backend = StaticSearch(_MAX_VALIDATE + 4)
    checked = []

    async def validate(url):
        checked.append(url)
        return FetchedPage(url=url, text="", ok=False)

    registry = ToolRegistry()
    registry.register(build_search_spec(FetchCache(), backend, validator=validate))
    result = run(
        dispatch(
            "search", {"queries": ["example"], "max_results": len(backend.hits)}, registry=registry
        )
    )
    assert result.structured["count"] == len(backend.hits)
    assert len(checked) == _MAX_VALIDATE
    assert f"Readability checks passed: 0 of {len(checked)}." in result.for_model()
