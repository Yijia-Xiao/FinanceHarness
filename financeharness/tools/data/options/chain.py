"""data.options.chain — listed option chain for one expiry (yfinance).

The expirations Yahoo lists for an underlying, and one expiry's calls and puts:
strike, bid/ask (+ mid), last price and trade time, volume, open interest,
Yahoo's implied volatility, and the in-the-money flag — plus the underlying spot
and the time to expiry (calendar days / 365) ready to chain into
compute.options.* (`prev:<id>.calls[*].strike`, `prev:<id>.calls[*].mid`).
Trimmed to the strikes nearest spot, since a full chain can run to hundreds of
rows. Yahoo quotes are delayed and carry no Greeks, and Yahoo's implied vol is
unreliable without a live bid/ask (e.g. outside market hours).
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import (
  PROVIDER,
  TICKER_MAX_LEN,
  df_cell,
  is_num,
  num,
  pct,
  yf_call,
)
from financeharness.tools.data.market.quotes import last_quote

_DESCRIPTION = (
    "Listed option chain for a US equity or ETF: the available expirations plus"
    " one expiry's calls and puts — strike, bid/ask/mid, last price, volume,"
    " open interest, Yahoo implied volatility, in-the-money flag — with the"
    " underlying spot and time_to_expiry_years ready to chain into"
    " compute.options.black_scholes / compute.options.implied_vol. expiration"
    " is YYYY-MM-DD (default: the first expiry at least min_days_to_expiry"
    " out); max_strikes keeps the strikes nearest spot. Yahoo data is delayed"
    " and has no Greeks; its implied vol is unreliable without a live bid/ask."
)
_DAYS_PER_YEAR = 365
_LISTED_SHOWN = 12

# our key -> yfinance option_chain column (numeric fields)
_FIELDS = {
    "strike": "strike",
    "bid": "bid",
    "ask": "ask",
    "last": "lastPrice",
    "volume": "volume",
    "open_interest": "openInterest",
    "implied_vol": "impliedVolatility",
}


class OptionsChainRequest(BaseModel):
  """Input for fetching one expiry of a listed option chain."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Underlying ticker (e.g. SPY, AAPL).",
  )
  expiration: str | None = Field(
      None,
      description=(
          "Expiry date YYYY-MM-DD from the listed expirations; omit for the"
          " first expiry at least min_days_to_expiry out."
      ),
  )
  min_days_to_expiry: int = Field(
      7,
      ge=0,
      le=3650,
      description=(
          "When expiration is omitted, skip expiries sooner than this many"
          " calendar days (default 7; near-zero tenors price poorly)."
      ),
  )
  max_strikes: int = Field(
      20,
      ge=1,
      le=500,
      description="Keep the N strikes nearest spot for calls and for puts.",
  )

  @field_validator("expiration")
  @classmethod
  def _ve(cls, v):
    if v is not None:
      try:
        datetime.date.fromisoformat(v)
      except ValueError as e:
        raise ValueError("expiration must be YYYY-MM-DD") from e
    return v


def _pick_expiry(expirations, today, min_days):
  """First listed expiry at least ``min_days`` out, else the last one listed."""
  for e in expirations:
    if (datetime.date.fromisoformat(e) - today).days >= min_days:
      return e
  return expirations[-1]


def _rows(df):
  """An option_chain frame → JSON-safe rows (NaN → None), with a bid/ask mid."""
  if df is None or getattr(df, "empty", True):
    return []
  out: list[dict[str, Any]] = []
  for _, row in df.iterrows():
    symbol = row.get("contractSymbol")
    rec: dict[str, Any] = {
        "contract": symbol if isinstance(symbol, str) else None
    }
    for key, col in _FIELDS.items():
      rec[key] = df_cell(row.get(col))
    if rec["strike"] is None:
      continue
    for key in ("volume", "open_interest"):
      if rec[key] is not None:
        rec[key] = int(rec[key])
    bid, ask = rec["bid"], rec["ask"]
    rec["mid"] = (
        round((bid + ask) / 2, 4) if bid and ask and 0 < bid <= ask else None
    )
    rec["in_the_money"] = bool(row.get("inTheMoney"))
    traded = row.get("lastTradeDate")
    rec["last_trade"] = (
        traded.strftime("%Y-%m-%d %H:%M UTC")
        if hasattr(traded, "strftime") and traded == traded
        else None
    )
    out.append(rec)
  return out


def _nearest(rows, spot, n):
  """The ``n`` rows with strikes closest to ``spot``, back in strike order."""
  keep = sorted(rows, key=lambda r: abs(r["strike"] - spot))[:n]
  return sorted(keep, key=lambda r: r["strike"])


def _quote_cell(r):
  if r is None:
    return "–", "–", "–", "–"
  oi = r["open_interest"]
  return (
      f"{num(r['bid'])}/{num(r['ask'])}",
      num(r["last"]),
      pct(r["implied_vol"]),
      str(oi) if oi is not None else "n/a",
  )


def _markdown(s):
  listed = s["expirations"]
  more = (
      f" … (+{len(listed) - _LISTED_SHOWN} more)"
      if len(listed) > _LISTED_SHOWN
      else ""
  )
  calls = {r["strike"]: r for r in s["calls"]}
  puts = {r["strike"]: r for r in s["puts"]}
  lines = [
      (
          f"**{s['ticker']} options** — expiry {s['expiration']}"
          f" ({s['days_to_expiry']}d, T={num(s['time_to_expiry_years'], 4)}y)"
          f" · spot {num(s['spot'])} · showing {len(s['calls'])} of"
          f" {s['n_calls_listed']} calls, {len(s['puts'])} of"
          f" {s['n_puts_listed']} puts nearest spot"
      ),
      f"Expirations ({len(listed)}): {', '.join(listed[:_LISTED_SHOWN])}{more}",
      "",
      "| Strike | Call bid/ask | Call last | Call IV | Call OI | Put bid/ask"
      " | Put last | Put IV | Put OI |",
      "|---|---|---|---|---|---|---|---|---|",
  ]
  for k in sorted(set(calls) | set(puts)):
    cells = (*_quote_cell(calls.get(k)), *_quote_cell(puts.get(k)))
    lines.append(f"| {num(k)} | " + " | ".join(cells) + " |")
  lines.append(
      "\nYahoo data: delayed, no Greeks; Yahoo IV is unreliable without a live"
      " bid/ask — recompute from mids with compute.options.implied_vol."
  )
  return "\n".join(lines)


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()
  tk = yf.Ticker(ticker)

  expirations = await yf_call(lambda: asyncio.to_thread(lambda: tk.options))
  expirations = list(expirations or ())
  if not expirations:
    raise ToolError(
        f"no listed options for '{ticker}' — check the ticker (Yahoo covers"
        " mainly US-listed equities, ETFs, and indices)"
    )
  today = datetime.date.today()
  expiry = req.expiration or _pick_expiry(
      expirations, today, req.min_days_to_expiry
  )
  if expiry not in expirations:
    raise ToolError(
        f"{expiry} is not a listed {ticker} expiry; listed:"
        f" {', '.join(expirations[:_LISTED_SHOWN])}"
    )

  chain = await yf_call(lambda: asyncio.to_thread(tk.option_chain, expiry))
  underlying = getattr(chain, "underlying", None) or {}
  spot = underlying.get("regularMarketPrice")
  if not is_num(spot):
    quote = await last_quote(ticker)
    spot = quote[0] if quote else None
  if not is_num(spot):
    raise ToolError(f"no spot price for '{ticker}' — provider may be down")
  spot = round(float(spot), 4)

  calls, puts = _rows(chain.calls), _rows(chain.puts)
  if not calls and not puts:
    raise ToolError(f"empty option chain for {ticker} {expiry}")
  days = (datetime.date.fromisoformat(expiry) - today).days
  structured = {
      "ticker": ticker,
      "spot": spot,
      "expiration": expiry,
      "days_to_expiry": days,
      # calendar-day basis, to the expiry date (0 on expiry day)
      "time_to_expiry_years": round(max(days, 0) / _DAYS_PER_YEAR, 6),
      "expirations": expirations,
      "n_calls_listed": len(calls),
      "n_puts_listed": len(puts),
      "calls": _nearest(calls, spot, req.max_strikes),
      "puts": _nearest(puts, spot, req.max_strikes),
  }
  return ToolResponse(
      markdown=_markdown(structured),
      structured=structured,
      meta={"provider": PROVIDER},
  )


SPEC = ToolSpec(
    name="data_options_chain",
    display_name="data.options.chain",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=OptionsChainRequest,
    handler=_handler,
    tags=("options", "derivatives", "chain", "implied-volatility", "data"),
)
