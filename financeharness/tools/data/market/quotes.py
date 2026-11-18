"""Shared quote helper for the market-data tools.

Last close + day change% for an index/yield symbol, via the equity tools'
yfinance retry wrapper.
"""

from __future__ import annotations

import asyncio

from financeharness.tools.data.equity.common import is_num, yf_call


async def last_quote(symbol):
    """(last close, day change %) for ``symbol`` from a short history window, or None

    if unavailable. Day change is vs the prior close (0.0 if only one bar).
    """
    import yfinance as yf

    def _sync():
        h = yf.Ticker(symbol).history(period="5d")
        if h is None or h.empty or "Close" not in h.columns:
            return None
        closes = [float(c) for c in h["Close"].tolist() if is_num(c)]
        return closes or None

    closes = await yf_call(lambda: asyncio.to_thread(_sync))
    if not closes:
        return None
    last = closes[-1]
    prev = closes[-2] if len(closes) > 1 else last
    change = (last - prev) / prev * 100 if prev else 0.0
    return round(last, 2), round(change, 2)
