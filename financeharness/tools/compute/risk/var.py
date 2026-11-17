"""compute.risk.var: Value-at-Risk of a return series (historical and parametric).

Pure-math. From a dated price series (the bars of data.equity.prices), the
worst expected loss at a confidence level over a horizon, in periods of the
return series: historical (empirical left-tail quantile) and parametric
(normal). Volatility scales with sqrt(horizon) and the parametric drift scales
linearly, so parametric VaR is -(mu*h + z*sigma*sqrt(h)). Reported as a
positive % loss. A downside-risk lens on a single name.
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
  joint_returns,
  unwrap_series,
)
from financeharness.tools.format import pct

_DESCRIPTION = (
    "Value-at-Risk for a dated price series (prices is prev:<call_id>.bars"
    " from data_equity_prices): the worst expected loss at a confidence level"
    " over a horizon, both historical (empirical tail) and parametric (normal)."
    " Returns positive % losses. confidence default 0.95, horizon"
    " default 1 (in periods of the series)."
)
_MIN_POINTS = 5


class VarRequest(BaseModel):
  """Input for historical value-at-risk over a price series."""

  prices: list[PricePoint] = Field(
      Ellipsis,
      min_length=_MIN_POINTS,
      description=f"Price series (5 or more points): {POINTS_DESCRIPTION}.",
  )
  price_field: PriceField = Field("adj_close", description=PRICE_FIELD_DESCRIPTION)

  _unwrap = field_validator("prices", mode="before")(unwrap_series)
  confidence: float = Field(
      0.95, gt=0.5, lt=1, description="Confidence level, e.g. 0.95."
  )
  horizon: int = Field(
      1,
      ge=1,
      le=252,
      description="Horizon in periods of the return series (>=1).",
  )


def _percentile(xs, q):
  s = sorted(xs)
  idx = q * (len(s) - 1)
  lo = int(idx)
  if lo + 1 >= len(s):
    return s[-1]
  return s[lo] * (1 - (idx - lo)) + s[lo + 1] * (idx - lo)


def compute_var(
    prices, confidence, horizon, price_field = "adj_close"
):
  """Compute VaR (historical + parametric) from percentage returns.

  Volatility scales with sqrt(horizon); the parametric drift scales linearly, so
  parametric VaR is -(mu*horizon + z*sigma*sqrt(horizon)). The historical
  quantile is sqrt-scaled (a common short-horizon approximation).
  """
  dated = dated_prices(prices, price_field, "prices")
  rets = joint_returns({"prices": list(dated.values())})["prices"]
  scale = horizon**0.5
  tail = 1 - confidence
  hist = -_percentile(rets, tail) * scale
  mu = statistics.fmean(rets)
  sigma = statistics.pstdev(rets)
  z = statistics.NormalDist().inv_cdf(tail)  # negative
  param = -(mu * horizon + z * sigma * scale)
  return {
      "confidence": confidence,
      "horizon": horizon,
      "historical_var_pct": round(max(hist, 0.0), 6),
      "parametric_var_pct": round(max(param, 0.0), 6),
      "mean_return": round(mu, 6),
      "volatility": round(sigma, 6),
      "n_obs": len(rets),
      "start": min(dated).isoformat(),
      "end": max(dated).isoformat(),
  }


def _markdown(res):
  return (
      f"**Value-at-Risk** ({pct(res['confidence'])} conf,"
      f" {res['horizon']} period(s), n={res['n_obs']}):\n  historical"
      f" {pct(res['historical_var_pct'])}, parametric"
      f" {pct(res['parametric_var_pct'])} loss\n  (per-period vol"
      f" {pct(res['volatility'])})"
  )


async def _handler(req):
  res = compute_var(req.prices, req.confidence, req.horizon, req.price_field)
  return ToolResponse(markdown=_markdown(res), structured=res, meta={})


SPEC = ToolSpec(
    name="compute_risk_var",
    display_name="compute.risk.var",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=VarRequest,
    handler=_handler,
    tags=("compute", "risk", "var", "downside"),
)
