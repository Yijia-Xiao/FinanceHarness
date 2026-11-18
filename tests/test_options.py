"""Option tools: Black-Scholes prices and Greeks, implied volatility, the option chain."""

import datetime
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.compute.options import OPTIONS_SPECS
from financeharness.tools.compute.options.black_scholes import bsm
from financeharness.tools.compute.options.implied_vol import implied_vol
from financeharness.tools.data.options import OPTIONS_DATA_SPECS
from financeharness.tools.data.options import chain as chain_module
from tests.support import run

TODAY = datetime.date(2026, 3, 2)
EXPIRATIONS = ["2026-03-02", "2026-03-06", "2026-03-20", "2026-06-19"]


@pytest.fixture(scope="module")
def registry():
    reg = ToolRegistry()
    for spec in (*OPTIONS_SPECS, *OPTIONS_DATA_SPECS):
        reg.register(spec)
    return reg


def call(registry, name, **args):
    result = run(dispatch(name, args, registry=registry))
    if result.ok:
        json.dumps(result.to_log(), allow_nan=False)  # every payload is strict JSON
    return result


def pricing_args(**over):
    args = {
        "spot": 100,
        "strikes": [100],
        "time_to_expiry_years": 1,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
    }
    args.update(over)
    return args


def test_price_and_greeks_match_the_textbook_case(registry):
    # S=K=100, T=1, r=5%, vol=20%, no dividend.
    out = call(registry, "compute_options_black_scholes", **pricing_args()).structured
    option = out["options"][0]
    assert option["d1"] == pytest.approx(0.35)
    assert option["d2"] == pytest.approx(0.15)
    assert option["call"]["price"] == pytest.approx(10.450584, abs=1e-5)
    assert option["put"]["price"] == pytest.approx(5.573526, abs=1e-5)
    assert option["call"]["delta"] == pytest.approx(0.636831, abs=1e-5)
    assert option["put"]["delta"] == pytest.approx(-0.363169, abs=1e-5)
    assert option["call"]["gamma"] == pytest.approx(0.018762, abs=1e-5)
    assert option["call"]["vega"] == pytest.approx(37.524035, abs=1e-4)
    assert option["call"]["theta"] == pytest.approx(-6.414028, abs=1e-4)
    assert option["put"]["theta"] == pytest.approx(-1.657880, abs=1e-4)
    assert option["call"]["rho"] == pytest.approx(53.232482, abs=1e-4)
    assert option["put"]["rho"] == pytest.approx(-41.890461, abs=1e-4)
    assert option["call"]["theta_per_day"] == pytest.approx(-6.414028 / 365, abs=1e-6)
    assert option["call"]["vega_per_point"] == pytest.approx(0.375240, abs=1e-5)


@pytest.mark.parametrize("strike", [60, 95, 100, 130])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_call_and_put_prices_satisfy_parity(strike, q):
    spot, t, r, sigma = 100.0, 0.5, 0.04, 0.3
    res = bsm(spot, strike, t, r, sigma, q)
    forward_gap = spot * math.exp(-q * t) - strike * math.exp(-r * t)
    assert res["call"]["price"] - res["put"]["price"] == pytest.approx(forward_gap, abs=1e-9)


def test_delta_and_vega_match_finite_differences():
    spot, strike, t, r, sigma, q = 100.0, 105.0, 0.25, 0.04, 0.25, 0.01
    res = bsm(spot, strike, t, r, sigma, q)
    h = 1e-4
    for leg in ("call", "put"):
        up, down = bsm(spot + h, strike, t, r, sigma, q), bsm(spot - h, strike, t, r, sigma, q)
        delta = (up[leg]["price"] - down[leg]["price"]) / (2 * h)
        assert res[leg]["delta"] == pytest.approx(delta, abs=1e-6)
        up, down = bsm(spot, strike, t, r, sigma + h, q), bsm(spot, strike, t, r, sigma - h, q)
        vega = (up[leg]["price"] - down[leg]["price"]) / (2 * h)
        assert res[leg]["vega"] == pytest.approx(vega, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.5])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_implied_vol_recovers_the_volatility_that_priced_the_option(option_type, sigma, strike):
    spot, t, r, q = 100.0, 0.5, 0.04, 0.01
    price = bsm(spot, strike, t, r, sigma, q)[option_type]["price"]
    vol, reason = implied_vol(option_type, price, spot, strike, t, r, q)
    if vol is None:
        # far out of the money at low volatility the price is below what a float resolves
        assert price < 1e-6, reason
    else:
        assert vol == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_tool_reports_each_strike_with_a_reason_when_unsolved(registry):
    fair = bsm(100.0, 100.0, 0.5, 0.04, 0.3)["call"]["price"]
    result = call(
        registry,
        "compute_options_implied_vol",
        option_type="call",
        spot=100,
        strikes=[100, 90, 100, 100],
        prices=[fair, 5.0, None, 150.0],
        time_to_expiry_years=0.5,
        risk_free_rate=0.04,
    )
    rows = result.structured["results"]
    assert rows[0]["implied_vol"] == pytest.approx(0.3, abs=1e-5)
    assert rows[1]["implied_vol"] is None and "intrinsic" in rows[1]["reason"]
    assert rows[2]["reason"] == "no price"
    assert rows[3]["implied_vol"] is None and "upper bound" in rows[3]["reason"]
    assert result.structured["n_solved"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_option_prices_are_rejected(registry, bad):
    result = call(
        registry,
        "compute_options_implied_vol",
        option_type="call",
        spot=100,
        strikes=[100],
        prices=[bad],
        time_to_expiry_years=0.5,
        risk_free_rate=0.04,
    )
    assert result.structured == {"error": "schema_validation"}


@pytest.mark.parametrize(
    "over",
    [
        {"spot": float("inf")},
        {"strikes": [float("nan")]},
        {"strikes": [0]},
        {"spot": 1e308, "strikes": [1e-308]},
        {"strikes": [1e12]},
        {"time_to_expiry_years": 0},
        {"volatility": 0},
        {"strikes": list(range(1, 102))},
    ],
)
def test_pricing_inputs_outside_their_bounds_are_rejected(registry, over):
    result = call(registry, "compute_options_black_scholes", **pricing_args(**over))
    assert result.structured == {"error": "schema_validation"}


def frame(strikes, bid, ask):
    return pd.DataFrame(
        {
            "contractSymbol": [f"TEST{int(k)}" for k in strikes],
            "strike": strikes,
            "bid": [bid] * len(strikes),
            "ask": [ask] * len(strikes),
            "lastPrice": [1.0] * len(strikes),
            "volume": [10.0] * len(strikes),
            "openInterest": [float("nan")] * len(strikes),
            "impliedVolatility": [0.25] * len(strikes),
            "inTheMoney": [k < 100 for k in strikes],
            "lastTradeDate": [pd.Timestamp("2026-02-27 20:59", tz="UTC")] * len(strikes),
        }
    )


class FakeTicker:
    """Stands in for ``yfinance.Ticker``: fixed expirations and a wide chain."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.options = tuple(EXPIRATIONS)
        self.requested = []

    def option_chain(self, expiry):
        self.requested.append(expiry)
        strikes = [float(k) for k in range(40, 161)]
        return SimpleNamespace(
            calls=frame(strikes, 1.0, 1.2),
            puts=frame(strikes, 0.0, 0.5),
            underlying={"regularMarketPrice": 100.4},
        )


@pytest.fixture
def market(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(chain_module, "_exchange_today", lambda: TODAY)


def test_chain_skips_an_expiry_dated_today_even_with_no_minimum(registry, market):
    out = call(registry, "data_options_chain", ticker="test", min_days_to_expiry=0).structured
    assert out["expiration"] == "2026-03-06"
    assert out["days_to_expiry"] == 4
    assert out["time_to_expiry_years"] == pytest.approx(4 / 365, abs=1e-6)


def test_chain_picks_the_first_expiry_past_the_minimum_and_trims_to_spot(registry, market):
    result = call(registry, "data_options_chain", ticker="test", max_strikes=5)
    out = result.structured
    assert out["ticker"] == "TEST"
    assert out["expiration"] == "2026-03-20"
    assert out["spot"] == 100.4
    assert [r["strike"] for r in out["calls"]] == [98.0, 99.0, 100.0, 101.0, 102.0]
    assert out["n_calls_listed"] == 121
    assert out["calls"][0]["mid"] == 1.1
    assert out["calls"][0]["open_interest"] is None
    assert out["puts"][0]["mid"] is None
    assert "showing 5 of 121 calls" in result.markdown


def test_chain_refuses_an_expiry_dated_today_and_lists_later_ones(registry, market):
    result = call(registry, "data_options_chain", ticker="TEST", expiration="2026-03-02")
    assert result.structured == {"error": "tool_error"}
    assert "expires today" in result.markdown
    assert "2026-03-06, 2026-03-20, 2026-06-19" in result.markdown


def test_chain_refuses_an_expiry_that_is_not_listed(registry, market):
    result = call(registry, "data_options_chain", ticker="TEST", expiration="2026-04-01")
    assert result.structured == {"error": "tool_error"}
    assert "not a listed TEST expiry" in result.markdown


def test_chain_never_returns_more_strikes_than_the_pricing_tools_accept(registry, market):
    too_many = call(registry, "data_options_chain", ticker="TEST", max_strikes=101)
    assert too_many.structured == {"error": "schema_validation"}

    chain = call(registry, "data_options_chain", ticker="TEST", max_strikes=100).structured
    priced = call(
        registry,
        "compute_options_black_scholes",
        spot=chain["spot"],
        strikes=[r["strike"] for r in chain["calls"]],
        time_to_expiry_years=chain["time_to_expiry_years"],
        risk_free_rate=0.04,
        volatility=0.25,
    )
    assert priced.ok
    assert len(priced.structured["options"]) == 100


def test_exchange_date_is_read_in_new_york_time():
    late_evening = datetime.datetime(2026, 3, 3, 2, 30, tzinfo=datetime.UTC)
    morning = datetime.datetime(2026, 3, 3, 14, 30, tzinfo=datetime.UTC)
    assert chain_module._exchange_today(late_evening) == datetime.date(2026, 3, 2)
    assert chain_module._exchange_today(morning) == datetime.date(2026, 3, 3)
