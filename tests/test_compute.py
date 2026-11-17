"""Compute tools: calc, cost of capital, DCF, DCF sensitivity, beta, correlation, VaR."""

import datetime
import json
import math

import pytest

from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolError, ToolRegistry
from financeharness.tools.compute import COMPUTE_SPECS
from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
from financeharness.tools.compute.arithmetic import safe_eval
from tests.support import run

Z_95 = 1.6448536269514722


@pytest.fixture(scope="module")
def registry():
  reg = ToolRegistry()
  reg.register(CALC_SPEC)
  for spec in COMPUTE_SPECS:
    reg.register(spec)
  return reg


def call(registry, name, **args):
  return run(dispatch(name, args, registry=registry))


def compound(start, returns):
  """Price series that realizes ``returns`` from ``start``."""
  prices = [start]
  for r in returns:
    prices.append(prices[-1] * (1 + r))
  return prices


RETURNS = [0.02, -0.01, 0.03, -0.02, 0.01]


def bars(prices, start="2026-03-02", step_days=1, field="adj_close"):
  """Dated points for ``prices``, one every ``step_days`` from ``start``."""
  first = datetime.date.fromisoformat(start)
  return [
      {"date": (first + datetime.timedelta(days=i * step_days)).isoformat(), field: p}
      for i, p in enumerate(prices)
  ]


def wave(n):
  """Returns that vary without a short repeating pattern."""
  return [0.01 * math.sin(1.7 * i) + 0.004 * math.cos(0.3 * i) for i in range(n)]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(37-13)/13*100", 184.6153846153846),
        ("2 + 3 * 4", 14),
        ("-(2 ** 10) // 3", -342),
        ("17 % 5", 2),
        ("round(sqrt(2), 3)", 1.414),
        ("max(1, abs(-7), min(9, 8))", 8),
        ("exp(log(5))", 5.0),
    ],
)
def test_calc_evaluates_arithmetic(expression, expected):
  assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "x + 1",
        "(1).real",
        "[1, 2][0]",
        "'a' * 3",
        "round(1.234, ndigits=1)",
        "1 +",
        "1 / 0",
        "sqrt(-1)",
        "2 ** 1001",
        "2 ** -1001",
    ],
)
def test_calc_refuses_anything_but_bounded_arithmetic(expression):
  with pytest.raises(ToolError):
    safe_eval(expression)


def test_calc_tool_returns_the_exact_value_and_a_short_display(registry):
  result = call(registry, "calc", expression="(37-13)/13*100")
  assert result.structured["result"] == pytest.approx(184.6153846153846)
  assert result.markdown == "`(37-13)/13*100` = **184.615**"


def test_calc_tool_turns_a_bad_expression_into_a_tool_error(registry):
  result = call(registry, "calc", expression="1 / 0")
  assert not result.ok
  assert result.structured == {"error": "tool_error"}
  assert "division by zero" in result.markdown


def test_calc_refuses_chained_powers_that_exceed_the_exponent_limit():
  with pytest.raises(ToolError):
    safe_eval("(10 ** 1000) ** 1000")


@pytest.mark.parametrize(
    "expression",
    ["(2 ** 1000) ** 16", "(2 ** 1000) ** 9", " * ".join(["2 ** 1000"] * 9)],
)
def test_calc_refuses_integer_results_beyond_the_size_limit(expression):
  with pytest.raises(ToolError, match="result too large"):
    safe_eval(expression)


def test_calc_integer_limit_is_8192_bits():
  assert safe_eval("(2 ** 1000) ** 8 * 2 ** 191").bit_length() == 8192
  with pytest.raises(ToolError, match="result too large"):
    safe_eval("(2 ** 1000) ** 8 * 2 ** 192")


def test_calc_shows_and_saves_the_largest_integers_it_accepts(registry):
  result = call(registry, "calc", expression="(2 ** 1000) ** 8")
  assert result.ok
  assert result.structured["result"] == 2**8000
  assert str(2**8000) in result.markdown
  json.dumps(result.to_log())


@pytest.mark.xfail(
    strict=True, reason="calc returns complex numbers, which cannot be saved in a trajectory"
)
def test_calc_result_can_always_be_saved_as_json(registry):
  result = call(registry, "calc", expression="(-8) ** (1 / 3)")
  json.dumps(result.to_log())


def test_cost_of_equity_follows_capm(registry):
  result = call(
      registry, "compute_valuation_wacc", risk_free_rate=0.04, equity_risk_premium=0.05, beta=1.2
  )
  assert result.structured["cost_of_equity"] == pytest.approx(0.10)
  assert result.structured["wacc"] == pytest.approx(0.10)
  assert result.structured["method"] == "CAPM only"


def test_wacc_blends_equity_with_after_tax_debt(registry):
  result = call(
      registry,
      "compute_valuation_wacc",
      risk_free_rate=0.04,
      equity_risk_premium=0.05,
      beta=1.2,
      cost_of_debt=0.06,
      tax_rate=0.25,
      debt_to_equity=0.5,
  )
  parts = result.structured["components"]
  assert parts["equity_weight"] == pytest.approx(2 / 3)
  assert parts["after_tax_cost_of_debt"] == pytest.approx(0.045)
  assert result.structured["wacc"] == pytest.approx(0.0816667, abs=1e-7)
  assert result.structured["wacc_premium_over_rf"] == pytest.approx(0.0416667, abs=1e-7)
  assert "**WACC = 8.2%**" in result.markdown


def test_wacc_needs_all_three_debt_inputs_or_none(registry):
  result = call(
      registry,
      "compute_valuation_wacc",
      risk_free_rate=0.04,
      equity_risk_premium=0.05,
      beta=1.2,
      cost_of_debt=0.06,
  )
  assert result.structured == {"error": "schema_validation"}
  assert "Got 1/3" in result.markdown


def test_dcf_with_gordon_growth_terminal_value(registry):
  result = call(
      registry,
      "compute_valuation_dcf",
      fcf_schedule=[100, 110],
      discount_rate=0.10,
      terminal_growth_rate=0.02,
      net_debt=40.909090909,
      shares_outstanding=100,
  )
  out = result.structured
  assert out["pv_fcfs"] == pytest.approx([90.9090909, 90.9090909])
  assert out["terminal_value"] == pytest.approx(1402.5)
  assert out["pv_terminal"] == pytest.approx(1159.0909091)
  assert out["enterprise_value"] == pytest.approx(1340.9090909)
  assert out["equity_value"] == pytest.approx(1300.0)
  assert out["intrinsic_per_share"] == pytest.approx(13.0)
  assert out["tv_share_of_ev"] == pytest.approx(1159.0909091 / 1340.9090909)


def test_dcf_with_exit_multiple_terminal_value(registry):
  result = call(
      registry,
      "compute_valuation_dcf",
      fcf_schedule=[100],
      discount_rate=0.25,
      terminal_method="exit_multiple",
      terminal_ebitda=50,
      terminal_ebitda_multiple=10,
      net_debt=80,
  )
  out = result.structured
  assert out["enterprise_value"] == pytest.approx(480.0)
  assert out["equity_value"] == pytest.approx(400.0)
  assert out["intrinsic_per_share"] is None
  assert "Intrinsic value/share" not in result.markdown


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"terminal_growth_rate": 0.10}, "must be < discount_rate"),
        ({}, "terminal_growth_rate is required"),
        ({"terminal_method": "exit_multiple", "terminal_ebitda": 50}, "both required"),
        ({"terminal_growth_rate": 0.02, "discount_rate": 1.5}, "discount_rate"),
        ({"terminal_growth_rate": 0.02, "fcf_schedule": []}, "fcf_schedule"),
    ],
)
def test_dcf_rejects_inputs_it_cannot_value(registry, args, message):
  base = {"fcf_schedule": [100, 110], "discount_rate": 0.10}
  result = call(registry, "compute_valuation_dcf", **{**base, **args})
  assert result.structured == {"error": "schema_validation"}
  assert message in result.markdown


def test_sensitivity_grid_values_each_cell_and_blanks_invalid_ones(registry):
  result = call(
      registry,
      "compute_valuation_dcf_sensitivity",
      fcf_schedule=[100, 110],
      discount_rates=[0.08, 0.10],
      terminal_growth_rates=[0.02, 0.09],
      shares_outstanding=10,
  )
  out = result.structured
  assert out["metric"] == "intrinsic_per_share"
  assert out["grid"][0] == [pytest.approx(179.0123457), None]
  assert out["grid"][1] == pytest.approx([134.0909091, 1009.0909091])
  assert out["low"] == pytest.approx(134.0909091)
  assert out["high"] == pytest.approx(1009.0909091)
  assert out["n_valid"] == 3


def test_sensitivity_grid_with_no_valid_cell_says_so(registry):
  result = call(
      registry,
      "compute_valuation_dcf_sensitivity",
      fcf_schedule=[100],
      discount_rates=[0.05],
      terminal_growth_rates=[0.05, 0.06],
  )
  assert result.ok
  assert result.structured["grid"] == [[None, None]]
  assert result.structured["low"] is None
  assert "No valid cells" in result.markdown


def test_sensitivity_grid_over_exit_multiples_reports_equity_value(registry):
  result = call(
      registry,
      "compute_valuation_dcf_sensitivity",
      fcf_schedule=[100],
      discount_rates=[0.25],
      terminal_method="exit_multiple",
      terminal_ebitda=50,
      terminal_ebitda_multiples=[10, 20],
      net_debt=80,
  )
  assert result.structured["metric"] == "equity_value"
  assert result.structured["grid"] == [pytest.approx([400.0, 800.0])]


def test_beta_recovers_the_sensitivity_to_the_benchmark(registry):
  benchmark = bars(compound(4500.0, RETURNS))
  asset = bars(compound(180.0, [0.002 + 1.5 * r for r in RETURNS]))
  result = call(registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=benchmark)
  assert result.structured == {
      "beta": 1.5,
      "correlation": 1.0,
      "r_squared": 1.0,
      "n_obs": 5,
      "start": "2026-03-02",
      "end": "2026-03-07",
      "dropped": {"asset": 0, "benchmark": 0},
  }


def test_beta_is_undefined_against_a_flat_benchmark(registry):
  result = call(
      registry,
      "compute_risk_beta",
      asset_prices=bars(compound(180.0, RETURNS)),
      benchmark_prices=bars([100.0] * 6),
  )
  assert result.ok
  assert result.structured["beta"] is None
  assert "n/a" in result.markdown


def test_beta_pairs_returns_by_date_when_sessions_are_missing(registry):
  market = wave(252)
  benchmark = bars(compound(4500.0, market))
  asset = bars(compound(180.0, [1.5 * r for r in market]))
  kept = [p for i, p in enumerate(asset) if i not in {20, 21, 60, 100, 101, 102, 180, 230}]

  out = call(registry, "compute_risk_beta", asset_prices=kept, benchmark_prices=benchmark)

  assert out.ok
  # Over a gap the asset compounds 1.5x each session, so its multi-session
  # return is close to, not exactly, 1.5x the benchmark's.
  assert out.structured["beta"] == pytest.approx(1.5, abs=0.01)
  assert out.structured["correlation"] == pytest.approx(1.0, abs=0.001)
  assert out.structured["n_obs"] == 244
  assert out.structured["dropped"] == {"asset": 0, "benchmark": 8}


def test_beta_ignores_the_order_of_the_points(registry):
  benchmark = bars(compound(4500.0, RETURNS))
  asset = bars(compound(180.0, [1.5 * r for r in RETURNS]))
  out = call(
      registry, "compute_risk_beta", asset_prices=asset[::-1], benchmark_prices=benchmark
  ).structured
  assert out["beta"] == 1.5


def test_beta_refuses_series_on_different_intervals(registry):
  market = wave(252)
  daily = bars(compound(180.0, [1.5 * r for r in market]))
  weekly = bars(compound(4500.0, market[:50]), step_days=7)

  result = call(registry, "compute_risk_beta", asset_prices=daily, benchmark_prices=weekly)

  assert result.structured == {"error": "tool_error"}
  assert "different bar intervals" in result.markdown
  assert "asset every 1 day(s), benchmark every 7 day(s)" in result.markdown


def test_short_series_across_a_weekend_is_still_daily(registry):
  market = wave(60)
  benchmark = bars(compound(4500.0, market), start="2026-01-05")
  # Friday, Monday, Tuesday: most gaps in so short a series are not one day.
  recent = [p for p in benchmark if p["date"] in {"2026-01-09", "2026-01-12", "2026-01-13"}]
  sparse = [p for p in benchmark if p["date"] in {"2026-01-05", "2026-01-12", "2026-01-13"}]
  for asset in (recent, sparse):
    out = call(registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=benchmark)
    assert out.ok
    assert out.structured["beta"] == pytest.approx(1.0)
    assert out.structured["n_obs"] == 2


def test_monthly_series_of_different_month_lengths_are_one_interval(registry):
  dates = ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"]
  a = [{"date": d, "adj_close": p} for d, p in zip(dates, [10, 11, 12, 11, 13], strict=True)]
  b = [{"date": d, "adj_close": p} for d, p in zip(dates[1:], [5, 6, 5, 7], strict=True)]
  out = call(registry, "compute_risk_correlation", series={"A": a, "B": b})
  assert out.ok
  assert out.structured["n_obs"] == 3


@pytest.mark.parametrize("price", [0, -4.0])
def test_prices_must_be_positive(registry, price):
  asset = bars(compound(180.0, RETURNS))
  asset[2]["adj_close"] = price
  beta = call(
      registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=bars([1, 2, 3, 4, 5, 6])
  )
  var = call(registry, "compute_risk_var", prices=asset)
  for result in (beta, var):
    assert result.structured == {"error": "schema_validation"}


def test_beta_refuses_series_with_too_few_common_dates(registry):
  asset = bars(compound(180.0, RETURNS), start="2026-03-02")
  benchmark = bars(compound(4500.0, RETURNS), start="2026-03-06")
  result = call(registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=benchmark)
  assert result.structured == {"error": "tool_error"}
  assert "share 2 date(s)" in result.markdown


def test_risk_tools_read_adj_close_unless_told_to_read_close(registry):
  market = compound(4500.0, RETURNS)
  traded = compound(180.0, [0.5 * r for r in RETURNS])
  adjusted = compound(180.0, [1.5 * r for r in RETURNS])
  dates = [p["date"] for p in bars(market)]
  asset = [
      {"date": d, "close": c, "adj_close": a, "volume": 10}
      for d, c, a in zip(dates, traded, adjusted, strict=True)
  ]
  benchmark = [{"date": d, "close": m, "adj_close": m} for d, m in zip(dates, market, strict=True)]

  default = call(registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=benchmark)
  as_traded = call(
      registry,
      "compute_risk_beta",
      asset_prices=asset,
      benchmark_prices=benchmark,
      price_field="close",
  )

  assert default.structured["beta"] == 1.5
  assert as_traded.structured["beta"] == 0.5


def test_point_without_the_requested_price_is_an_input_error(registry):
  asset = bars(compound(180.0, RETURNS), field="close")
  benchmark = bars(compound(4500.0, RETURNS))
  result = call(registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=benchmark)
  assert result.structured == {"error": "tool_error"}
  assert "asset: the point dated 2026-03-02 has no adj_close" in result.markdown


def test_two_points_on_one_date_are_an_input_error(registry):
  asset = bars(compound(180.0, RETURNS))
  asset[1]["date"] = asset[0]["date"]
  result = call(
      registry, "compute_risk_beta", asset_prices=asset, benchmark_prices=bars([1, 2, 3, 4, 5, 6])
  )
  assert result.structured == {"error": "tool_error"}
  assert "more than one point dated 2026-03-02" in result.markdown


@pytest.mark.parametrize(
    "args",
    [
        {"asset_prices": [180, 182, 179], "benchmark_prices": [4500, 4520, 4490]},
        {"asset_prices": [{"adj_close": 1}] * 3, "benchmark_prices": [{"adj_close": 1}] * 3},
    ],
)
def test_beta_does_not_accept_prices_without_dates(registry, args):
  result = call(registry, "compute_risk_beta", **args)
  assert result.structured == {"error": "schema_validation"}


def test_correlation_matrix_is_symmetric_with_exact_extremes(registry):
  series = {
      "UP": bars(compound(100.0, RETURNS)),
      "SAME": bars(compound(20.0, RETURNS)),
      "MIRROR": bars(compound(50.0, [0.01 - 2 * r for r in RETURNS])),
      "FLAT": bars([10.0] * 6),
  }
  out = call(registry, "compute_risk_correlation", series=series).structured
  matrix = out["matrix"]
  assert out["names"] == ["UP", "SAME", "MIRROR", "FLAT"]
  assert out["n_obs"] == 5
  assert matrix["UP"]["UP"] == 1.0
  assert matrix["UP"]["SAME"] == matrix["SAME"]["UP"] == 1.0
  assert matrix["UP"]["MIRROR"] == matrix["MIRROR"]["UP"] == -1.0
  assert matrix["UP"]["FLAT"] is None
  assert matrix["FLAT"]["FLAT"] is None


def test_correlation_of_partly_related_series_matches_the_hand_value(registry):
  series = {
      "A": bars(compound(100.0, [0.01, 0.02, 0.03])),
      "B": bars(compound(100.0, [0.01, 0.03, 0.02])),
  }
  out = call(registry, "compute_risk_correlation", series=series).structured
  assert out["matrix"]["A"]["B"] == 0.5


def test_correlation_uses_only_the_dates_every_series_has(registry):
  market = wave(60)
  a = bars(compound(100.0, market))
  b = bars(compound(50.0, market))
  c = bars(compound(20.0, [-r for r in market]))
  del b[10:13]
  del c[40]

  result = call(registry, "compute_risk_correlation", series={"A": a, "B": b, "C": c})
  out = result.structured

  assert out["matrix"]["A"]["B"] == pytest.approx(1.0, abs=0.001)
  assert out["matrix"]["A"]["C"] == pytest.approx(-1.0, abs=0.01)
  assert out["n_obs"] == 56
  assert out["dropped"] == {"A": 4, "B": 1, "C": 3}
  assert (out["start"], out["end"]) == (a[0]["date"], a[-1]["date"])
  assert "points without a common date: A 4, B 1, C 3" in result.markdown


def test_correlation_refuses_series_on_different_intervals(registry):
  series = {"D": bars(compound(100.0, wave(60))), "W": bars(compound(100.0, wave(12)), step_days=7)}
  result = call(registry, "compute_risk_correlation", series=series)
  assert result.structured == {"error": "tool_error"}
  assert "different bar intervals" in result.markdown


def test_risk_tools_reject_series_that_are_too_short(registry):
  beta = call(
      registry,
      "compute_risk_beta",
      asset_prices=bars([1, 2]),
      benchmark_prices=bars([1, 2, 3]),
  )
  series = {"A": bars([1, 2, 3]), "B": bars([1, 2])}
  corr = call(registry, "compute_risk_correlation", series=series)
  var = call(registry, "compute_risk_var", prices=bars([1, 2, 3, 4]))
  for result in (beta, corr, var):
    assert result.structured == {"error": "schema_validation"}


def test_var_reports_historical_and_parametric_loss(registry):
  prices = bars(compound(100.0, [0.03, -0.05, 0.01, -0.02, 0.04]))
  sigma = 0.001096**0.5

  one = call(registry, "compute_risk_var", prices=prices).structured
  four = call(registry, "compute_risk_var", prices=prices, horizon=4).structured

  assert one["n_obs"] == 5
  assert (one["start"], one["end"]) == ("2026-03-02", "2026-03-07")
  assert one["mean_return"] == pytest.approx(0.002)
  assert one["volatility"] == pytest.approx(sigma, abs=1e-6)
  assert one["historical_var_pct"] == pytest.approx(0.044)
  assert one["parametric_var_pct"] == pytest.approx(Z_95 * sigma - 0.002, abs=1e-6)
  assert four["historical_var_pct"] == pytest.approx(0.088)
  assert four["parametric_var_pct"] == pytest.approx(2 * Z_95 * sigma - 0.008, abs=1e-6)


def test_var_is_zero_when_the_tail_is_a_gain(registry):
  out = call(registry, "compute_risk_var", prices=bars([100, 101, 103, 104, 106, 109])).structured
  assert out["historical_var_pct"] == 0.0


def test_var_orders_the_points_by_date(registry):
  prices = bars(compound(100.0, [0.03, -0.05, 0.01, -0.02, 0.04]))
  ordered = call(registry, "compute_risk_var", prices=prices).structured
  shuffled = call(registry, "compute_risk_var", prices=prices[3:] + prices[:3]).structured
  assert shuffled == ordered


def test_var_without_any_usable_return_is_an_input_error(registry):
  result = call(registry, "compute_risk_var", prices=bars([0, 0, 0, 0, 0]))
  assert result.structured == {"error": "schema_validation"}
