"""Page fetching: request headers per host."""

import pytest

from financeharness.providers.profiles import ModelProfile
from financeharness.runtime.config import RuntimeConfig, load_runtime_config
from financeharness.tools.research import FetchCache, assembly, visit
from financeharness.tools.research.visit_fetch import _FETCH_HEADERS, FetchedPage, _headers_for
from tests.support import run, runtime_config


@pytest.mark.parametrize(
    "url",
    [
        "https://www.sec.gov/Archives/edgar/data/320193/a10-k.htm",
        "https://sec.gov/cgi-bin/browse-edgar",
        "https://data.SEC.gov/submissions/CIK0000320193.json",
    ],
)
def test_sec_gov_requests_carry_the_declared_agent(url):
  headers = _headers_for(url, "Example Research admin@example.com")
  assert headers["User-Agent"] == "Example Research admin@example.com"
  assert headers["Accept"] == _FETCH_HEADERS["Accept"]


@pytest.mark.parametrize(
    "url",
    ["https://example.com/10k", "https://notsec.gov/x", "https://sec.gov.example.com/x", "x"],
)
def test_other_hosts_keep_the_browser_agent(url):
  assert _headers_for(url, "Example Research") == _FETCH_HEADERS


def test_declared_agent_defaults_to_the_runtime_config_value():
  headers = _headers_for("https://www.sec.gov/")
  assert headers["User-Agent"] == RuntimeConfig().fetch.sec_user_agent


def test_declared_agent_is_set_in_the_file_and_overridden_by_the_environment(tmp_path):
  path = tmp_path / "runtime.json"
  path.write_text('{"fetch": {"sec_user_agent": "From File"}}', encoding="utf-8")
  assert load_runtime_config(path, env={}).fetch.sec_user_agent == "From File"
  env = {"FH_SEC_USER_AGENT": "From Env admin@example.com"}
  assert load_runtime_config(path, env=env).fetch.sec_user_agent == "From Env admin@example.com"


@pytest.mark.parametrize("bad", ["", "   ", "Recherche \u00e9quipe", "two\nlines"])
def test_unusable_agent_strings_keep_the_default(tmp_path, bad):
  default = RuntimeConfig().fetch.sec_user_agent
  path = tmp_path / "absent.json"
  assert load_runtime_config(path, env={"FH_SEC_USER_AGENT": bad}).fetch.sec_user_agent == default
  with pytest.raises(ValueError, match="printable ASCII"):
    RuntimeConfig(fetch={"sec_user_agent": bad})


def test_visit_and_search_fetch_with_the_configured_agent(tmp_path, monkeypatch):
  seen = []

  async def fake_fetch(url, *, sec_user_agent, **_kwargs):
    seen.append(sec_user_agent)
    return FetchedPage(url=url, text="", ok=False, error="offline")

  monkeypatch.setattr(visit, "http_fetch", fake_fetch)
  monkeypatch.setattr(assembly, "quick_fetch", fake_fetch)
  captured = {}
  real_search_spec = assembly.build_search_spec

  def capture(cache, backend, validator=None):
    captured["validator"] = validator
    return real_search_spec(cache, backend, validator=validator)

  monkeypatch.setattr(assembly, "build_search_spec", capture)
  config = runtime_config(tmp_path, fetch={"sec_user_agent": "Example Research"})
  registry = assembly.build_research_registry(
      FetchCache(),
      ModelProfile(name="reader", model="m"),
      config=config,
  )

  result = run(registry.get("visit").handler(visit.VisitRequest(urls=["https://www.sec.gov/x"])))
  run(captured["validator"]("https://www.sec.gov/y"))

  assert seen == ["Example Research", "Example Research"]
  assert result.structured["pages"][0]["error"] == "offline"

