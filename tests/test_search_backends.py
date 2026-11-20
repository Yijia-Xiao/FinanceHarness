"""Search backends: the You.com backend's requests and result mapping, and backend selection."""

import httpx
import pytest

from financeharness.runtime.config import load_runtime_config
from financeharness.tools.research.search_backends import (
    DdgsBackend,
    YouComBackend,
    default_backend,
)
from tests.support import run

_BODY = {
    "results": {
        "web": [
            {
                "title": " Apple 10-K ",
                "url": "https://example.com/10k",
                "description": "short",
                "snippets": ["Revenue grew", "Margins held"],
            },
            {"title": "no url", "url": "", "description": "skipped"},
            {"title": "Fallback", "url": "https://example.org/", "description": "from description"},
        ]
    }
}


class _Calls(list):
    """Requests seen by the fake transport, plus the response to give."""

    status = 200
    body = _BODY


@pytest.fixture
def calls(monkeypatch):
    """Route the backend's httpx client to a recorder; no network."""
    seen = _Calls()

    def handler(request):
        seen.append(request)
        return httpx.Response(seen.status, json=seen.body)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    return seen


def test_keyless_search_uses_the_free_endpoint_and_maps_results(calls, monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    hits = run(YouComBackend().search("apple revenue", 5))
    assert [h.url for h in hits] == ["https://example.com/10k", "https://example.org/"]
    assert hits[0].title == "Apple 10-K"
    assert hits[0].snippet == "Revenue grew Margins held"
    assert hits[1].snippet == "from description"
    req = calls[0]
    assert req.method == "GET"
    assert str(req.url).startswith("https://api.you.com/v1/agents/search?")
    assert req.url.params["query"] == "apple revenue"
    assert req.url.params["count"] == "5"
    assert "x-api-key" not in req.headers
    assert req.headers["user-agent"] == "youdotcom-integration/Yijia-Xiao-FinanceHarness"


def test_keyless_count_is_capped_at_fifty(calls, monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    run(YouComBackend().search("q", 80))
    assert calls[0].url.params["count"] == "50"


def test_keyed_search_posts_with_the_key(calls, monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "secret")
    hits = run(YouComBackend().search("apple revenue", 80))
    assert len(hits) == 2
    req = calls[0]
    assert req.method == "POST"
    assert str(req.url) == "https://ydc-index.io/v1/search"
    assert req.headers["x-api-key"] == "secret"
    assert req.headers["user-agent"] == "youdotcom-integration/Yijia-Xiao-FinanceHarness"
    assert req.read() == b'{"query":"apple revenue","count":80}'


def test_http_errors_raise_so_the_search_tool_can_skip_the_query(calls, monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    calls.status = 429
    with pytest.raises(httpx.HTTPStatusError):
        run(YouComBackend().search("q", 5))


def test_ddgs_stays_the_default_backend():
    assert isinstance(default_backend(), DdgsBackend)


def test_backend_is_chosen_in_the_file_or_the_environment(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text('{"search": {"backend": "youcom"}}', encoding="utf-8")
    assert isinstance(default_backend(load_runtime_config(path, env={})), YouComBackend)
    cfg = load_runtime_config(path, env={"FH_SEARCH_BACKEND": "ddgs"})
    assert isinstance(default_backend(cfg), DdgsBackend)


def test_unknown_backend_in_the_environment_keeps_the_lower_layer():
    assert load_runtime_config(env={"FH_SEARCH_BACKEND": "nope"}).search.backend == "ddgs"
