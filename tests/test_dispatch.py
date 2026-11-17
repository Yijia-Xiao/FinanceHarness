"""Tool dispatch: argument repair, prev: reference resolution, validation, error mapping."""

import sys

import pytest
from pydantic import BaseModel

from financeharness.runtime.chaining import resolve_one, resolve_references
from financeharness.runtime.dispatch import dispatch, dispatch_json_args
from financeharness.runtime.tool_registry import (
  ToolError,
  ToolRegistry,
  ToolSessionState,
)
from financeharness.tools.compute import COMPUTE_SPECS
from tests.support import echo_handler, make_tool, run


class SeriesRequest(BaseModel):
  values: list[float]
  label: str = ""


class GridRequest(BaseModel):
  rows: list[list[float]]


class WideRequest(BaseModel):
  a: int
  b: int
  c: int
  d: int
  e: int
  f: int
  g: int


class Recorder:
  """Handler that counts its calls and can be told to fail."""

  def __init__(self, error=None):
    self.calls = 0
    self.error = error

  async def __call__(self, req):
    self.calls += 1
    if self.error is not None:
      raise self.error
    return echo_handler(req)


@pytest.fixture
def series_handler():
  return Recorder()


@pytest.fixture
def registry(series_handler):
  reg = ToolRegistry()
  reg.register(make_tool("series", SeriesRequest, series_handler))
  reg.register(make_tool("grid", GridRequest, Recorder()))
  reg.register(make_tool("wide", WideRequest, Recorder()))
  for spec in COMPUTE_SPECS:
    reg.register(spec)
  return reg


PRICES = {
    "ticker": "ACME",
    "bars": [
        {"close": 10.0, "ohlc": [9.0, 11.0]},
        {"close": 11.0, "ohlc": [10.0, 12.0]},
        {"close": 12.1, "ohlc": [11.0, 13.0]},
    ],
    "by_year": {2024: 5.0},
    "matrix": [[1.0, 2.0], [3.0, 4.0]],
}


@pytest.fixture
def state():
  return ToolSessionState(tool_results={"p1": PRICES})


def test_unknown_tool_is_reported_with_the_visible_tools(registry):
  result = run(dispatch("nope", {}, registry=registry))
  assert not result.ok
  assert result.structured == {"error": "unknown_tool"}
  assert "'nope'" in result.markdown
  assert "'series'" in result.markdown


def test_dotted_display_name_reaches_the_tool(registry):
  result = run(
      dispatch(
          "compute.valuation.wacc",
          {"risk_free_rate": 0.04, "equity_risk_premium": 0.05, "beta": 1.0},
          registry=registry,
      )
  )
  assert result.ok
  assert result.structured["wacc"] == pytest.approx(0.09)


def test_stringified_list_is_parsed_for_a_list_field(registry):
  result = run(dispatch("series", {"values": "[1, 2, 3]"}, registry=registry))
  assert result.ok
  assert result.structured["values"] == [1.0, 2.0, 3.0]


def test_string_field_that_looks_like_json_is_left_alone(registry):
  result = run(dispatch("series", {"values": [1], "label": "[1, 2]"}, registry=registry))
  assert result.structured["label"] == "[1, 2]"


def test_unparseable_stringified_list_fails_validation_on_that_field(registry, series_handler):
  result = run(dispatch("series", {"values": "[1, 2"}, registry=registry))
  assert not result.ok
  assert result.structured == {"error": "schema_validation"}
  assert "values:" in result.markdown
  assert series_handler.calls == 0


def test_validation_message_lists_at_most_five_issues(registry):
  result = run(dispatch("wide", {}, registry=registry))
  assert result.structured == {"error": "schema_validation"}
  assert result.markdown.count("Field required") == 5
  assert "2 more" in result.markdown


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("prev:p1", PRICES),
        ("prev:p1.ticker", "ACME"),
        ("prev:p1.bars[0].close", 10.0),
        ("prev:p1.bars[-1].close", 12.1),
        ("prev:p1.bars[*].close", [10.0, 11.0, 12.1]),
        ("prev:p1.bars[*].ohlc[1]", [11.0, 12.0, 13.0]),
        ("prev:p1.bars[1].ohlc", [10.0, 12.0]),
        ("prev:p1.matrix[*][0]", [1.0, 3.0]),
        ("prev:p1.by_year.2024", 5.0),
    ],
)
def test_reference_resolves_against_an_earlier_result(reference, expected):
  assert resolve_one(reference, {"p1": PRICES}) == expected


@pytest.mark.parametrize(
    ("reference", "reason"),
    [
        ("prev:", "missing call_id"),
        ("prev:zz.bars", "known: ['p1']"),
        ("prev:p1.volume", "available: ['bars', 'by_year', 'matrix', 'ticker']"),
        ("prev:p1.bars[3].close", "index out of range (length 3)"),
        ("prev:p1.ticker[0]", "requires a list, got str"),
        ("prev:p1.ticker.name", "expected dict, got str"),
        ("prev:p1.matrix[*][*]", "nested '[*]'"),
        ("prev:p1.bars[first]", "unsupported indexer"),
        ("prev:p1..bars", "empty path segment"),
    ],
)
def test_unresolvable_reference_explains_what_is_wrong(reference, reason):
  error = resolve_one(reference, {"p1": PRICES})
  assert error.reference == reference
  assert reason in error.reason


def test_references_are_replaced_at_any_depth_and_other_values_kept():
  args = {
      "series": {"ACME": "prev:p1.bars[*].close", "FLAT": [1.0, 1.0]},
      "labels": ["prev:p1.ticker", "plain"],
      "window": 3,
  }
  resolved, errors = resolve_references(args, {"p1": PRICES})
  assert errors == []
  assert resolved == {
      "series": {"ACME": [10.0, 11.0, 12.1], "FLAT": [1.0, 1.0]},
      "labels": ["ACME", "plain"],
      "window": 3,
  }


def test_reference_wrapped_in_a_list_expands_once_for_a_flat_list_field(registry, state):
  args = {"values": ["prev:p1.bars[*].close"]}
  result = run(dispatch("series", args, registry=registry, session_state=state))
  assert result.ok
  assert result.structured["values"] == [10.0, 11.0, 12.1]


def test_reference_wrapped_in_a_list_stays_one_row_for_a_nested_list_field(registry, state):
  args = {"rows": ["prev:p1.bars[*].close"]}
  result = run(dispatch("grid", args, registry=registry, session_state=state))
  assert result.ok
  assert result.structured["rows"] == [[10.0, 11.0, 12.1]]


def test_failed_reference_stops_the_call_before_the_handler(registry, state, series_handler):
  args = {"values": "prev:p1.bars[*].volume"}
  result = run(dispatch("series", args, registry=registry, session_state=state))
  assert not result.ok
  assert result.structured["error"] == "chain_resolution"
  assert [f["reference"] for f in result.structured["failures"]] == ["prev:p1.bars[*].volume"]
  assert "prev:p1.bars[*].volume" in result.markdown
  assert result.args == args
  assert series_handler.calls == 0


def test_malformed_index_in_a_reference_is_reported_not_raised(registry, state):
  args = {"values": "prev:p1.bars[--5]"}
  result = run(dispatch("series", args, registry=registry, session_state=state))
  assert not result.ok
  assert result.structured["error"] == "chain_resolution"


def test_unexpected_error_while_preparing_arguments_is_reported_not_raised(
    registry, state, series_handler, monkeypatch
):
  def broken(args, tool_results):
    raise RuntimeError("boom")

  module = sys.modules["financeharness.runtime.dispatch"]
  monkeypatch.setattr(module, "resolve_references", broken)
  args = {"values": "prev:p1.bars[*].close"}
  result = run(dispatch("series", args, registry=registry, session_state=state))
  assert not result.ok
  assert result.structured == {"error": "argument_error"}
  assert "RuntimeError: boom" in result.markdown
  assert result.args == args
  assert series_handler.calls == 0


@pytest.mark.parametrize("args_json", ["[1, 2]", '"text"', "7", "null"])
def test_arguments_that_are_not_a_json_object_are_reported_not_raised(registry, state, args_json):
  result = run(dispatch_json_args("series", args_json, registry=registry, session_state=state))
  assert result.structured["error"] in {"schema_validation", "argument_error"}


def test_arguments_nested_too_deeply_to_parse_are_reported_not_raised(registry, state):
  args_json = '{"values": ' + "[" * 100_000 + "]" * 100_000 + "}"
  result = run(dispatch_json_args("series", args_json, registry=registry, session_state=state))
  assert result.structured == {"error": "args_not_json"}


def test_result_of_one_call_feeds_a_compute_tool(registry):
  state = ToolSessionState()
  benchmark = [100.0, 102.0, 101.0, 104.0, 103.0]
  asset = [50.0]
  for prev, cur in zip(benchmark, benchmark[1:], strict=False):
    asset.append(asset[-1] * (1 + 1.5 * (cur / prev - 1)))

  for call_id, values in (("c1", asset), ("c2", benchmark)):
    stored = run(
        dispatch(
            "series", {"values": values}, registry=registry, session_state=state, call_id=call_id
        )
    )
    assert f"prev:{call_id}." in stored.markdown

  result = run(
      dispatch(
          "compute_risk_beta",
          {"asset_prices": "prev:c1.values", "benchmark_prices": "prev:c2.values"},
          registry=registry,
          session_state=state,
          call_id="c3",
      )
  )
  assert result.ok
  assert result.structured["beta"] == pytest.approx(1.5)
  assert state.tool_results["c3"] == result.structured


def test_result_is_not_stored_or_footed_without_a_call_id(registry):
  state = ToolSessionState()
  result = run(dispatch("series", {"values": [1]}, registry=registry, session_state=state))
  assert result.markdown == "ok"
  assert state.tool_results == {}


def test_expected_tool_failure_passes_its_message_through():
  reg = ToolRegistry()
  reg.register(make_tool("series", SeriesRequest, Recorder(ToolError("no data for ACME"))))
  state = ToolSessionState()
  result = run(
      dispatch("series", {"values": [1]}, registry=reg, session_state=state, call_id="c1")
  )
  assert not result.ok
  assert result.markdown == "no data for ACME"
  assert result.structured == {"error": "tool_error"}
  assert state.tool_results == {}


def test_unexpected_handler_error_is_returned_with_its_type():
  reg = ToolRegistry()
  reg.register(make_tool("series", SeriesRequest, Recorder(KeyError("close"))))
  result = run(dispatch("series", {"values": [1]}, registry=reg))
  assert not result.ok
  assert result.structured == {"error": "handler_exception"}
  assert result.markdown.startswith("KeyError")
  assert result.meta["exception"] == "KeyError"


def test_log_entry_carries_the_call_and_its_structured_result(registry):
  result = run(dispatch("series", {"values": [1, 2]}, registry=registry))
  entry = result.to_log()
  assert entry["name"] == "series"
  assert entry["args"] == {"values": [1, 2]}
  assert entry["ok"] is True
  assert entry["structured"]["values"] == [1.0, 2.0]
  assert entry["markdown_chars"] == len(result.for_model())


def test_json_arguments_are_parsed_before_dispatch(registry):
  result = run(dispatch_json_args("series", '{"values": [1, 2]}', registry=registry))
  assert result.ok
  assert result.structured["values"] == [1.0, 2.0]


def test_arguments_that_are_not_json_are_reported(registry, series_handler):
  result = run(dispatch_json_args("series", '{"values": [1, 2', registry=registry))
  assert not result.ok
  assert result.structured == {"error": "args_not_json"}
  assert result.args == {"_raw": '{"values": [1, 2'}
  assert series_handler.calls == 0


def test_empty_argument_string_means_no_arguments(registry):
  result = run(dispatch_json_args("series", "", registry=registry))
  assert result.structured == {"error": "schema_validation"}
  assert "values: Field required" in result.markdown


@pytest.mark.parametrize("token", ["Infinity", "NaN"])
def test_non_finite_numbers_are_rejected_before_a_compute_tool_runs(registry, token):
  args = f'{{"fcf_schedule": [100, {token}], "discount_rate": 0.1, "terminal_growth_rate": 0.02}}'
  result = run(dispatch_json_args("compute_valuation_dcf", args, registry=registry))
  assert not result.ok
