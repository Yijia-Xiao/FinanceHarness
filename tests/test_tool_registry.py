"""Tool registry and two-tier loading: core tools are always offered, deferred ones on request."""

import json
import re

import pytest
from pydantic import BaseModel

from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolRegistry, ToolSessionState, ToolSpec
from financeharness.tools.core import build_load_tool_spec
from financeharness.tools.research import FetchCache, build_equity_research_registry
from tests.support import chat_profile, echo_handler, make_tool, run, runtime_config

WIRE_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class Empty(BaseModel):
    pass


async def _handler(req):
    return echo_handler(req)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(make_tool("always", Empty, _handler))
    reg.register(make_tool("data_b", Empty, _handler, tier="deferred"))
    reg.register(make_tool("data_a", Empty, _handler, tier="deferred"))
    return reg


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


def test_spec_rejects_a_dotted_wire_name():
    with pytest.raises(ValueError, match="underscores"):
        make_tool("data.equity", Empty, _handler)


def test_spec_rejects_a_description_too_short_to_route_on():
    with pytest.raises(ValueError, match="minimum 30"):
        ToolSpec(
            name="t",
            display_name="t",
            tier="core",
            description="too short",
            request_schema=Empty,
            handler=_handler,
        )


def test_registering_a_name_twice_fails(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool("always", Empty, _handler))


def test_clone_is_independent_of_its_source(registry):
    clone = registry.clone()
    clone.register(make_tool("extra", Empty, _handler))
    assert "extra" in clone.names()
    assert "extra" not in registry.names()


def test_catalog_lists_deferred_tools_only_in_name_order(registry):
    lines = registry.catalog_text().splitlines()
    assert [line.split("`")[1] for line in lines] == ["data_a", "data_b"]
    assert lines[0].startswith("- data.a (`data_a`): ")


def test_deferred_tool_is_offered_only_after_it_is_loaded(registry):
    state = ToolSessionState()
    assert _names(state.visible_schemas(registry)) == ["always"]

    loaded, unknown = state.load(["data_b", "always", "missing"], registry)

    assert [s.name for s in loaded] == ["data_b", "always"]
    assert unknown == ["missing"]
    assert _names(state.visible_schemas(registry)) == ["always", "data_b"]


def test_loading_is_per_run(registry):
    ToolSessionState().load(["data_a"], registry)
    assert _names(ToolSessionState().visible_schemas(registry)) == ["always"]


def test_load_tool_returns_schemas_and_names_unknown_tools(registry):
    state = ToolSessionState()
    registry.register(build_load_tool_spec(state, registry))

    result = run(dispatch("load_tool", {"tools": ["data.a", "nope"]}, registry=registry))

    assert result.ok
    assert result.structured == {"loaded": ["data_a"], "unknown": ["nope"]}
    assert "### `data_a`" in result.markdown
    assert "['data_a', 'data_b']" in result.markdown
    assert state.loaded_deferred == {"data_a"}


def test_load_tool_requires_at_least_one_name(registry):
    registry.register(build_load_tool_spec(ToolSessionState(), registry))
    result = run(dispatch("load_tool", {"tools": []}, registry=registry))
    assert result.structured == {"error": "schema_validation"}


@pytest.fixture
def product_registry(tmp_path):
    return build_equity_research_registry(
        FetchCache(), chat_profile(), config=runtime_config(tmp_path)
    )


def test_product_registry_keeps_data_and_compute_tools_deferred(product_registry):
    core = {s.name for s in product_registry.core_tools()}
    deferred = {s.name for s in product_registry.deferred_tools()}
    assert core == {"search", "visit", "compose_citations", "calc", "update_plan"}
    assert deferred
    assert all(name.startswith(("data_", "compute_")) for name in deferred)


def test_every_product_tool_has_a_valid_name_and_a_serializable_schema(product_registry):
    for name in product_registry.names():
        schema = product_registry.get(name).openai_schema()
        assert WIRE_NAME.match(schema["function"]["name"])
        assert json.loads(json.dumps(schema)) == schema
        assert schema["function"]["parameters"]["type"] == "object"
