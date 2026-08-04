"""compute.risk.var — Value-at-Risk of a return series (historical + parametric).

Pure-math. From a price series (chain from data.equity.prices), the worst
expected
loss at a confidence level over a horizon, in periods of the return series:
historical (empirical left-tail quantile) and parametric (normal). Volatility
scales with √horizon and the parametric drift scales linearly, so parametric VaR
is −(μ·h + zσ·√h). Reported as a positive % loss. A downside-risk lens on a
single name.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.numeric import Finite
from financeharness.tools.compute.risk.returns import pct_returns
from financeharness.tools.format import pct

_DESCRIPTION = (
    "Value-at-Risk for a price series (pass prices, e.g. from"
    " data.equity.prices): the worst expected loss at a confidence level over a"
    " horizon, both historical (empirical tail) and parametric (normal)."
    " Returns positive % losses. confidence default 0.95, horizon"
    " default 1 (in periods of the series)."
)
_MIN_POINTS = 5


class VarRequest(BaseModel):
  """Input for historical value-at-risk over a price series."""

  prices: list[Finite] = Field(
      Ellipsis,
      min_length=_MIN_POINTS,
      description="Price series (>=5), e.g. [180, 182, 179, 185, 183].",
  )
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
    prices, confidence, horizon
):
  """Compute VaR (historical + parametric) from percentage returns.

  Volatility scales with sqrt(horizon); the parametric drift scales linearly, so
  parametric VaR is -(mu*horizon + z*sigma*sqrt(horizon)). The historical
  quantile is sqrt-scaled (a common short-horizon approximation).
  """
  rets = pct_returns(prices)
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
  }


def _markdown(res):
  return (
      f"**Value-at-Risk** ({pct(res['confidence'])} conf,"
      f" {res['horizon']} period(s), n={res['n_obs']}):\n  historical"
      f" {pct(res['historical_var_pct'])} · parametric"
      f" {pct(res['parametric_var_pct'])} loss\n  (per-period vol"
      f" {pct(res['volatility'])})"
  )


async def _handler(req):
  res = compute_var(req.prices, req.confidence, req.horizon)
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
