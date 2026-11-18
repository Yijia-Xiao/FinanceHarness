"""compute.risk.beta: beta of an asset against a benchmark, from dated prices.

Pure-math. Joins the asset and benchmark series on date, converts the common
dates to returns, and computes beta = cov(asset, bench) / var(bench), plus the
correlation and R^2. A window-explicit beta beside the static one in
data.equity.reference.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field, field_validator

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.risk.returns import (
    POINTS_DESCRIPTION,
    PRICE_FIELD_DESCRIPTION,
    PriceField,
    PricePoint,
    dated_prices,
    join_on_date,
    joint_returns,
    unwrap_series,
)
from financeharness.tools.format import num

_DESCRIPTION = (
    "Beta of an asset against a benchmark from two dated price series (asset_prices"
    " and benchmark_prices, each prev:<call_id>.bars from data_equity_prices, fetched"
    " with the same period and interval; the benchmark is an index or index"
    " fund). Joins the series on date, so missing sessions do not shift the"
    " pairing, and refuses series on different intervals. Returns beta ="
    " cov/var(benchmark), correlation, R^2, the number of returns used and"
    " the common date range."
)
_MIN_POINTS = 3


class BetaRequest(BaseModel):
    """Input for estimating asset beta versus a benchmark series."""

    asset_prices: list[PricePoint] = Field(
        Ellipsis,
        min_length=_MIN_POINTS,
        description=f"Asset series: {POINTS_DESCRIPTION}.",
    )
    benchmark_prices: list[PricePoint] = Field(
        Ellipsis,
        min_length=_MIN_POINTS,
        description=f"Benchmark series: {POINTS_DESCRIPTION}.",
    )
    price_field: PriceField = Field("adj_close", description=PRICE_FIELD_DESCRIPTION)

    _unwrap = field_validator("asset_prices", "benchmark_prices", mode="before")(unwrap_series)


def compute_beta(asset, benchmark, price_field="adj_close"):
    """Compute beta from the returns of two dated series over their common dates."""
    dates, prices, dropped = join_on_date(
        {
            "asset": dated_prices(asset, price_field, "asset"),
            "benchmark": dated_prices(benchmark, price_field, "benchmark"),
        },
        _MIN_POINTS,
    )
    rets = joint_returns(prices)
    ra, rb = rets["asset"], rets["benchmark"]
    var_b = statistics.variance(rb) if len(rb) >= 2 else 0.0
    beta = statistics.covariance(ra, rb) / var_b if var_b else None
    corr = statistics.correlation(ra, rb) if var_b and statistics.variance(ra) else None
    return {
        "beta": round(beta, 4) if beta is not None else None,
        "correlation": round(corr, 4) if corr is not None else None,
        "r_squared": round(corr**2, 4) if corr is not None else None,
        "n_obs": len(ra),
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "dropped": dropped,
    }


def _markdown(res):
    dropped = res["dropped"]
    return (
        f"**Beta** (n={res['n_obs']}, {res['start']} to {res['end']}):"
        f" {num(res['beta'])}, correlation {num(res['correlation'])}, R^2"
        f" {num(res['r_squared'])}\n  points without a common date: asset"
        f" {dropped['asset']}, benchmark {dropped['benchmark']}"
    )


async def _handler(req):
    res = compute_beta(req.asset_prices, req.benchmark_prices, req.price_field)
    return ToolResponse(markdown=_markdown(res), structured=res, meta={})


SPEC = ToolSpec(
    name="compute_risk_beta",
    display_name="compute.risk.beta",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=BetaRequest,
    handler=_handler,
    tags=("compute", "risk", "beta", "capm"),
)
