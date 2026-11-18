"""Shared helpers for the equity data tools (yfinance implementation).

A single async wrapper runs the blocking yfinance call in a thread under the
runtime's transient-retry policy; provider-tagging + money formatting live here
too. yfinance is flaky (rate limits, transport blips): transient shapes retry,
everything else (bad ticker) surfaces as an actionable tool error.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Awaitable, Callable
from typing import Any

from financeharness.runtime.retry import is_default_retryable, with_retry
from financeharness.tools.format import money, num, pct  # re-exported for equity tools

__all__ = [
    "PROVIDER",
    "TICKER_MAX_LEN",
    "df_cell",
    "is_num",
    "is_transient_yf_error",
    "money",
    "num",
    "pct",
    "yf_call",
]

PROVIDER = "yfinance"
# Max length for a ticker request field: permissive enough for suffixed foreign
# listings (e.g. 005930.KS, RDSA.AS); shared so every equity tool agrees.
TICKER_MAX_LEN = 24


def is_num(v: Any) -> bool:
    """True for a real, finite number: the guard before any arithmetic.

    yfinance returns ``float('nan')`` for absent fields (and numpy scalars from
    its DataFrames), which must not flow into math or the JSON-serialized
    structured payload. ``numbers.Real`` covers both Python and numpy numerics;
    bool is excluded.
    """
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def df_cell(v: Any) -> float | None:
    """A DataFrame cell as a float, or ``None`` when it's NaN/absent.

    The JSON-safe coercion for yfinance frames (a raw NaN must not reach the
    structured payload).
    """
    return float(v) if is_num(v) else None


def is_transient_yf_error(err: BaseException) -> bool:
    """Classify yfinance/provider failures worth retrying."""

    name = type(err).__name__
    if name in {"HTTPError", "ConnectionError", "Timeout", "ReadTimeout"}:
        return True
    text = str(err).lower()
    return "rate" in text or "429" in text or is_default_retryable(err)


async def yf_call[T](factory: Callable[[], Awaitable[T]]) -> T:
    """Run an async yfinance attempt under transient retry.

    ``factory`` returns a fresh awaitable each call (wrap the blocking call with
    ``asyncio.to_thread``).
    """
    return await with_retry(factory, retryable=is_transient_yf_error)
