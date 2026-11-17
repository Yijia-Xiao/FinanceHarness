"""compute.risk.beta — beta of an asset vs a benchmark, from price series.

Pure-math. From an asset price series and a benchmark series (e.g. a stock vs
data.market.indices' S&P 500), converts to returns, aligns, and computes
β = cov(asset, bench) / var(bench), plus the correlation and R². A transparent,
window-explicit beta to complement the static one in data.equity.reference.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field, model_validator

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.compute.numeric import Finite
from financeharness.tools.compute.risk.returns import align, pct_returns
from financeharness.tools.format import num

_DESCRIPTION = (
    "Beta of an asset vs a benchmark from price series (asset_prices +"
    " benchmark_prices, e.g. a stock vs an index from data.market.indices)."
    " Converts to returns, aligns the common window, and returns beta ="
    " cov/var(benchmark), with correlation and R^2. A window-explicit beta vs"
    " the static one in reference."
)
_MIN_POINTS = 3


class BetaRequest(BaseModel):
  """Input for estimating asset beta versus a benchmark series."""

  asset_prices: list[Finite] = Field(
      Ellipsis,
      min_length=_MIN_POINTS,
      description="Asset price series, e.g. [180, 182, 179, 185].",
  )
  benchmark_prices: list[Finite] = Field(
      Ellipsis,
      min_length=_MIN_POINTS,
      description=(
          "Benchmark (e.g. index) price series, e.g. [4500, 4520, 4490, 4550]."
      ),
  )

  @model_validator(mode="after")
  def _check(self):
    if (
        len(self.asset_prices) < _MIN_POINTS
        or len(self.benchmark_prices) < _MIN_POINTS
    ):
      raise ValueError(f"both series need >= {_MIN_POINTS} prices")
    return self


def compute_beta(
    asset_prices, benchmark_prices
):
  """Compute beta from aligned asset and benchmark percentage returns."""

  ra, rb = align(pct_returns(asset_prices), pct_returns(benchmark_prices))
  if len(ra) < 2:
    raise ToolError("not enough overlapping observations to estimate beta")
  var_b = statistics.variance(rb)
  beta = statistics.covariance(ra, rb) / var_b if var_b else None
  corr = (
      statistics.correlation(ra, rb)
      if var_b and statistics.variance(ra)
      else None
  )
  return {
      "beta": round(beta, 4) if beta is not None else None,
      "correlation": round(corr, 4) if corr is not None else None,
      "r_squared": round(corr**2, 4) if corr is not None else None,
      "n_obs": len(ra),
  }


def _markdown(res):
  return (
      f"**Beta** (n={res['n_obs']}): {num(res['beta'])} · "
      f"correlation {num(res['correlation'])} · R² {num(res['r_squared'])}"
  )


async def _handler(req):
  res = compute_beta(req.asset_prices, req.benchmark_prices)
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
