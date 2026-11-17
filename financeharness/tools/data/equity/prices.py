"""data.equity.prices — OHLCV history + derived technicals.

`bars[*]` is the stable chaining surface (`prev:<id>.bars[*].adj_close` into
compute tools). Each bar's `close` is the price as traded (Yahoo adjusts it for
splits only); `adj_close` is also adjusted for dividends. Quoted prices are
as traded; the technicals run on `adj_close`, so a dividend's price drop isn't
read as a loss. Derived measures are named canon only — SMA(20/50/200),
golden/death cross, realized volatility (per-bar stdev + annualized to the bar
interval, factor shown). No interpretive tier labels; the model owns
interpretation.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec
from financeharness.tools.data.equity.common import PROVIDER, TICKER_MAX_LEN, is_num, num, yf_call

_VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
_VALID_INTERVALS = ("1d", "5d", "1wk", "1mo")
# Bars per year by interval — annualizes realized volatility correctly for
# non-daily bars (1d ≈ 252 trading days, 5d ≈ 50, 1wk = 52, 1mo = 12).
_BARS_PER_YEAR = {"1d": 252, "5d": 50, "1wk": 52, "1mo": 12}
_RECENT_ROWS = 8

_DESCRIPTION = (
    "OHLCV price history for a public equity plus derived technicals:"
    " SMA(20/50/200), golden/death cross, realized volatility (annualized to the"
    " bar interval), and the period high/low range. Each bar has close (the"
    " price as traded, adjusted for splits only) and adj_close (also adjusted"
    " for dividends): quote close for a price on a date, and chain"
    " bars[*].adj_close into return-based compute tools."
)


class EquityPricesRequest(BaseModel):
  """Input for fetching equity OHLCV history and technical measures."""

  ticker: str = Field(
      Ellipsis,
      min_length=1,
      max_length=TICKER_MAX_LEN,
      description="Equity ticker (e.g. NVDA).",
  )
  period: str = Field(
      "1y", description=f"Window: one of {', '.join(_VALID_PERIODS)}."
  )
  interval: str = Field(
      "1d", description=f"Bar interval: one of {', '.join(_VALID_INTERVALS)}."
  )

  @field_validator("period")
  @classmethod
  def _vp(cls, v):
    if v not in _VALID_PERIODS:
      raise ValueError(f"period must be one of {_VALID_PERIODS}")
    return v

  @field_validator("interval")
  @classmethod
  def _vi(cls, v):
    if v not in _VALID_INTERVALS:
      raise ValueError(f"interval must be one of {_VALID_INTERVALS}")
    return v


def _sma_last(values, window):
  return sum(values[-window:]) / window if len(values) >= window else None


def _sma_series(values, window):
  out: list[float | None] = []
  running = 0.0
  for i, v in enumerate(values):
    running += v
    if i >= window:
      running -= values[i - window]
    out.append(running / window if i + 1 >= window else None)
  return out


def _realized_vol(closes, bars_per_year):
  if len(closes) < 2:
    return None, None
  rets = []
  for i in range(1, len(closes)):
    a, b = closes[i - 1], closes[i]
    if a <= 0 or b <= 0:
      return None, None
    rets.append(math.log(b / a))
  n = len(rets)
  mean = sum(rets) / n
  var = sum((x - mean) ** 2 for x in rets) / (n - 1) if n > 1 else 0.0
  per_bar = math.sqrt(var)
  return per_bar, per_bar * math.sqrt(bars_per_year)


def _last_cross(closes):
  """Most recent SMA50 vs SMA200 crossover: 'golden' (50 over 200) or 'death'."""
  s50, s200 = _sma_series(closes, 50), _sma_series(closes, 200)
  last: str | None = None
  for i in range(1, len(closes)):
    if None in (s50[i], s50[i - 1], s200[i], s200[i - 1]):
      continue
    if s50[i - 1] <= s200[i - 1] and s50[i] > s200[i]:
      last = "golden"
    elif s50[i - 1] >= s200[i - 1] and s50[i] < s200[i]:
      last = "death"
  return last


def _build(ticker, rows, interval):
  closes = [r["close"] for r in rows]
  adj = [r["adj_close"] for r in rows]
  bars_per_year = _BARS_PER_YEAR.get(interval, 252)
  per_bar, annualized = _realized_vol(adj, bars_per_year)
  structured = {
      "ticker": ticker,
      "bars": rows,
      "current": closes[-1],
      "sma": {
          "20": _sma_last(adj, 20),
          "50": _sma_last(adj, 50),
          "200": _sma_last(adj, 200),
      },
      "last_cross": _last_cross(adj),
      "realized_vol": {
          "per_bar": per_bar,
          "annualized": annualized,
          "periods_per_year": bars_per_year,
      },
      "high": max(r["high"] for r in rows),
      "low": min(r["low"] for r in rows),
      "n_bars": len(rows),
  }
  sma = structured["sma"]
  recent = "\n".join(
      f"  {r['date']}: {num(r['close'])} (adj {num(r['adj_close'])})"
      for r in rows[-_RECENT_ROWS:]
  )
  cross = f" · {structured['last_cross']} cross" if structured["last_cross"] else ""
  md = (
      f"**{ticker}** — {len(rows)} bars · current"
      f" {num(closes[-1])}\nSMA20/50/200:"
      f" {num(sma['20'])}/{num(sma['50'])}/{num(sma['200'])}{cross}\nRealized"
      f" vol: per-bar {num(per_bar, 4)}, annualized {num(annualized, 3)}"
      f" (×√{bars_per_year}) · range"
      f" {num(structured['low'])}–{num(structured['high'])}\nRecent"
      f" closes as traded (dividend-adjusted):\n{recent}"
  )
  return ToolResponse(
      markdown=md, structured=structured, meta={"provider": PROVIDER}
  )


async def _handler(req):
  import yfinance as yf

  ticker = req.ticker.upper()

  def _sync():
    # auto_adjust=False keeps Close as traded and adds Adj Close; yfinance's
    # default rewrites Close with dividend-adjusted values.
    h = yf.Ticker(ticker).history(
        period=req.period, interval=req.interval, auto_adjust=False
    )
    rows: list[dict[str, Any]] = []
    for idx, row in h.iterrows():
      # yfinance emits NaN-filled bars (halts, sparse intervals); skip them so
      # int(NaN) can't crash and NaN can't poison technicals / the payload.
      if not all(
          is_num(row[c])
          for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
      ):
        continue
      rows.append({
          "date": idx.strftime("%Y-%m-%d"),
          "open": round(float(row["Open"]), 4),
          "high": round(float(row["High"]), 4),
          "low": round(float(row["Low"]), 4),
          "close": round(float(row["Close"]), 4),
          "adj_close": round(float(row["Adj Close"]), 4),
          "volume": int(row["Volume"]),
      })
    return rows

  rows = await yf_call(lambda: asyncio.to_thread(_sync))
  if not rows:
    raise ToolError(f"no price data for '{ticker}' — check the ticker / period")
  return _build(ticker, rows, req.interval)


SPEC = ToolSpec(
    name="data_equity_prices",
    display_name="data.equity.prices",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=EquityPricesRequest,
    handler=_handler,
    tags=("equity", "prices", "technical", "data"),
)
