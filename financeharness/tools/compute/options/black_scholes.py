"""compute.options.black_scholes — European option price + Greeks (BSM).

Pure-math. Black-Scholes-Merton for calls and puts on an underlying with a
continuous dividend yield q, at one or more strikes and a single volatility:

  d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ√T),  d2 = d1 − σ√T
  C = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
  P = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)

Strikes chain from data.options.chain (`prev:<id>.calls[*].strike`); a flat
volatility (e.g. realized vol from data.equity.prices) gives the model price to
set against the market's. Theta is per year (and per calendar day, ÷365); vega
and rho are per 1.00 change in σ / r (and per point, ÷100). European exercise
only — no early-exercise premium.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.format import num, pct

_DESCRIPTION = (
    "European option price + Greeks (Black-Scholes-Merton, continuous dividend"
    " yield) for calls and puts at one or more strikes: price, delta, gamma,"
    " theta (per year and per day), vega and rho (per 1.00 and per point)."
    " Pass spot, strikes (e.g. chained from data.options.chain),"
    " time_to_expiry_years, risk_free_rate (e.g. the 3-month yield from"
    " data.market.rates), volatility (e.g. realized vol from data.equity.prices)"
    " and optional dividend_yield — rates and vol as decimals (0.2 not 20)."
    " European exercise only: no early-exercise premium for American options."
)
_MAX_STRIKES = 100
_DAYS_PER_YEAR = 365


class BlackScholesRequest(BaseModel):
  """Input for Black-Scholes-Merton pricing at one or more strikes."""

  spot: float = Field(Ellipsis, gt=0, description="Underlying price, e.g. 100.")
  strikes: list[float] = Field(
      Ellipsis,
      min_length=1,
      max_length=_MAX_STRIKES,
      description="Strike(s) > 0, e.g. [95, 100, 105].",
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
  volatility: float = Field(
      Ellipsis, gt=0, le=5, description="Annualized volatility, decimal (0.2)."
  )
  dividend_yield: float = Field(
      0.0, ge=0, lt=1, description="Continuous dividend yield, decimal (0.015)."
  )

  @field_validator("strikes")
  @classmethod
  def _positive_strikes(cls, v):
    if any(k <= 0 for k in v):
      raise ValueError("every strike must be > 0")
    return v


def norm_cdf(x):
  """Standard normal CDF via erfc (accurate in the left tail)."""
  return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x):
  return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bsm(spot, strike, t, r, sigma, q=0.0):
  """Black-Scholes-Merton call/put price and Greeks for one strike.

  Greeks are in natural units: theta per year, vega per 1.00 of σ, rho per
  1.00 of r.
  """
  sqrt_t = math.sqrt(t)
  vol_t = sigma * sqrt_t
  d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
  d2 = d1 - vol_t
  disc_r, disc_q = math.exp(-r * t), math.exp(-q * t)
  pdf = norm_pdf(d1)
  nd1, nd2 = norm_cdf(d1), norm_cdf(d2)
  nmd1, nmd2 = norm_cdf(-d1), norm_cdf(-d2)
  decay = -spot * disc_q * pdf * sigma / (2.0 * sqrt_t)
  gamma = disc_q * pdf / (spot * vol_t)
  vega = spot * disc_q * pdf * sqrt_t
  call = {
      "price": spot * disc_q * nd1 - strike * disc_r * nd2,
      "delta": disc_q * nd1,
      "theta": decay - r * strike * disc_r * nd2 + q * spot * disc_q * nd1,
      "rho": strike * t * disc_r * nd2,
  }
  put = {
      "price": strike * disc_r * nmd2 - spot * disc_q * nmd1,
      "delta": -disc_q * nmd1,
      "theta": decay + r * strike * disc_r * nmd2 - q * spot * disc_q * nmd1,
      "rho": -strike * t * disc_r * nmd2,
  }
  for leg in (call, put):
    leg["gamma"] = gamma
    leg["vega"] = vega
  return {"d1": d1, "d2": d2, "call": call, "put": put}


def _leg(g):
  return {
      "price": round(g["price"], 6),
      "delta": round(g["delta"], 6),
      "gamma": round(g["gamma"], 6),
      "theta": round(g["theta"], 6),
      "theta_per_day": round(g["theta"] / _DAYS_PER_YEAR, 6),
      "vega": round(g["vega"], 6),
      "vega_per_point": round(g["vega"] / 100, 6),
      "rho": round(g["rho"], 6),
      "rho_per_point": round(g["rho"] / 100, 6),
  }


def compute_black_scholes(req):
  """Price every strike at the request's spot, tenor, rates, and volatility."""
  options = []
  for k in req.strikes:
    res = bsm(
        req.spot,
        k,
        req.time_to_expiry_years,
        req.risk_free_rate,
        req.volatility,
        req.dividend_yield,
    )
    options.append({
        "strike": k,
        "d1": round(res["d1"], 6),
        "d2": round(res["d2"], 6),
        "call": _leg(res["call"]),
        "put": _leg(res["put"]),
    })
  return {
      "model": "black-scholes-merton (european)",
      "inputs": {
          "spot": req.spot,
          "time_to_expiry_years": req.time_to_expiry_years,
          "risk_free_rate": req.risk_free_rate,
          "volatility": req.volatility,
          "dividend_yield": req.dividend_yield,
      },
      "options": options,
  }


def _markdown(res):
  i = res["inputs"]
  lines = [
      f"**Black-Scholes-Merton** (European) · S={num(i['spot'])} ·"
      f" T={num(i['time_to_expiry_years'], 4)}y · r={pct(i['risk_free_rate'], 2)}"
      f" · σ={pct(i['volatility'])} · q={pct(i['dividend_yield'], 2)}"
  ]
  for o in res["options"]:
    c, p = o["call"], o["put"]
    lines.append(
        f"  K {num(o['strike'])}: call {num(c['price'], 4)} (Δ"
        f" {num(c['delta'], 3)}, Θ/day {num(c['theta_per_day'], 4)}) · put"
        f" {num(p['price'], 4)} (Δ {num(p['delta'], 3)}, Θ/day"
        f" {num(p['theta_per_day'], 4)}) · Γ {num(c['gamma'], 4)} · vega/pt"
        f" {num(c['vega_per_point'], 4)}"
    )
  return "\n".join(lines)


async def _handler(req):
  res = compute_black_scholes(req)
  return ToolResponse(
      markdown=_markdown(res), structured=res, meta={"model": res["model"]}
  )


SPEC = ToolSpec(
    name="compute_options_black_scholes",
    display_name="compute.options.black_scholes",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=BlackScholesRequest,
    handler=_handler,
    tags=("compute", "options", "black-scholes", "greeks", "pricing"),
)
