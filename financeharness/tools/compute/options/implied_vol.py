"""compute.options.implied_vol — implied volatility from option prices (BSM).

Pure-math. Inverts Black-Scholes-Merton for the volatility that reproduces each
quoted price, by bisection: the price is monotone in σ, so inside the
no-arbitrage bounds the solve always converges. A price at or below intrinsic
value, or at/above the upper bound (discounted spot for a call, discounted
strike for a put), has no implied vol and comes back null with the reason.
Prices chain from data.options.chain (`prev:<id>.calls[*].mid`) to read the
smile across strikes; set it against Yahoo's own implied_vol and the realized
vol from data.equity.prices.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.options.black_scholes import bsm
from financeharness.tools.format import num, pct

_DESCRIPTION = (
    "Implied volatility by inverting Black-Scholes-Merton (bisection) for one"
    " option type at one or more strikes: pass option_type, spot, strikes and"
    " the matching prices (e.g. chained bid/ask mids from data.options.chain;"
    " null entries are skipped), time_to_expiry_years, risk_free_rate and"
    " optional dividend_yield (decimals). Returns implied_vol per strike as a"
    " decimal, or null with the reason when the price breaks the no-arbitrage"
    " bounds (e.g. a stale last trade below intrinsic value)."
)
_MAX_STRIKES = 100
_VOL_LO, _VOL_HI = 1e-6, 5.0
_VOL_TOL = 1e-8
_MAX_ITER = 200


class ImpliedVolRequest(BaseModel):
  """Input for solving implied volatility from option prices."""

  option_type: Literal["call", "put"] = Field(
      Ellipsis, description="call or put (applies to every strike)."
  )
  spot: float = Field(Ellipsis, gt=0, description="Underlying price, e.g. 100.")
  strikes: list[float] = Field(
      Ellipsis,
      min_length=1,
      max_length=_MAX_STRIKES,
      description="Strike(s) > 0, e.g. [95, 100, 105].",
  )
  prices: list[float | None] = Field(
      Ellipsis,
      min_length=1,
      max_length=_MAX_STRIKES,
      description=(
          "Option price per strike, same order and length as strikes (null to"
          " skip), e.g. [7.1, 4.25, 2.05]."
      ),
  )
  time_to_expiry_years: float = Field(
      Ellipsis,
      gt=0,
      le=10,
      description="Time to expiry in years, e.g. 30/365 = 0.0822.",
  )
  risk_free_rate: float = Field(
      Ellipsis,
      gt=-0.1,
      lt=1,
      description="Annual risk-free rate, decimal (0.045).",
  )
  dividend_yield: float = Field(
      0.0, ge=0, lt=1, description="Continuous dividend yield, decimal (0.015)."
  )

  @model_validator(mode="after")
  def _check_pairs(self):
    if len(self.prices) != len(self.strikes):
      raise ValueError(
          f"prices ({len(self.prices)}) must match strikes"
          f" ({len(self.strikes)}) one-to-one"
      )
    if any(k <= 0 for k in self.strikes):
      raise ValueError("every strike must be > 0")
    return self


def implied_vol(option_type, price, spot, strike, t, r, q=0.0):
  """(σ, None) where BSM(σ) reproduces ``price``, or (None, reason).

  Bisection on σ in [1e-6, 5]: the price is strictly increasing in σ, so the
  bracket halves to within 1e-8 in about 30 steps.
  """
  fwd_spot, pv_strike = spot * math.exp(-q * t), strike * math.exp(-r * t)
  if option_type == "call":
    lower, upper = max(fwd_spot - pv_strike, 0.0), fwd_spot
  else:
    lower, upper = max(pv_strike - fwd_spot, 0.0), pv_strike
  if price <= lower:
    return None, "price at or below intrinsic value (no time value)"
  if price >= upper:
    return None, "price at or above the no-arbitrage upper bound"

  def model(sigma):
    return bsm(spot, strike, t, r, sigma, q)[option_type]["price"]

  lo, hi = _VOL_LO, _VOL_HI
  if model(lo) > price:
    return None, "time value too small to resolve"
  if model(hi) < price:
    return None, f"implied vol above {pct(_VOL_HI, 0)}"
  for _ in range(_MAX_ITER):
    mid = 0.5 * (lo + hi)
    if model(mid) < price:
      lo = mid
    else:
      hi = mid
    if hi - lo < _VOL_TOL:
      break
  return 0.5 * (lo + hi), None


def compute_implied_vol(req):
  """Solve each (strike, price) pair; unsolvable ones carry a reason."""
  results = []
  for k, p in zip(req.strikes, req.prices, strict=True):
    row = {"strike": k, "price": p, "implied_vol": None}
    if p is None:
      row["reason"] = "no price"
    else:
      vol, reason = implied_vol(
          req.option_type,
          p,
          req.spot,
          k,
          req.time_to_expiry_years,
          req.risk_free_rate,
          req.dividend_yield,
      )
      if vol is None:
        row["reason"] = reason
      else:
        row["implied_vol"] = round(vol, 6)
    results.append(row)
  return {
      "option_type": req.option_type,
      "model": "black-scholes-merton (european)",
      "inputs": {
          "spot": req.spot,
          "time_to_expiry_years": req.time_to_expiry_years,
          "risk_free_rate": req.risk_free_rate,
          "dividend_yield": req.dividend_yield,
      },
      "results": results,
      "n_solved": sum(1 for r in results if r["implied_vol"] is not None),
  }


def _markdown(res):
  i = res["inputs"]
  lines = [
      f"**Implied volatility** ({res['option_type']}s, BSM) ·"
      f" S={num(i['spot'])} · T={num(i['time_to_expiry_years'], 4)}y ·"
      f" r={pct(i['risk_free_rate'], 2)} · q={pct(i['dividend_yield'], 2)}"
  ]
  for r in res["results"]:
    iv = pct(r["implied_vol"]) if r["implied_vol"] is not None else "n/a"
    why = f" ({r['reason']})" if r.get("reason") else ""
    lines.append(f"  K {num(r['strike'])} @ {num(r['price'], 4)} → {iv}{why}")
  lines.append(f"Solved {res['n_solved']}/{len(res['results'])}.")
  return "\n".join(lines)


async def _handler(req):
  res = compute_implied_vol(req)
  return ToolResponse(
      markdown=_markdown(res), structured=res, meta={"model": res["model"]}
  )


SPEC = ToolSpec(
    name="compute_options_implied_vol",
    display_name="compute.options.implied_vol",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=ImpliedVolRequest,
    handler=_handler,
    tags=("compute", "options", "implied-volatility", "pricing"),
)
