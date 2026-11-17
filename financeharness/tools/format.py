"""Canonical value formatters for tool output — dollars, percents, plain numbers.

One shared place so data and compute tools render figures identically. Negatives
are
rendered (a negative free cash flow shows as ``-$1.20B``, not ``n/a``);
``None``/NaN
(and bools) render as ``n/a``.
"""

from __future__ import annotations

import math


def _is_real(v):
  """A real, finite number — excludes None, NaN, infinity, and bools."""
  if isinstance(v, bool) or not isinstance(v, int | float):
    return False
  try:
    return math.isfinite(v)
  except OverflowError:  # an int too large to represent as a float
    return False


def money(v):
  """Compact dollar amount with a magnitude suffix; renders negatives."""
  if not _is_real(v):
    return "n/a"
  sign, a = ("-" if v < 0 else ""), abs(v)
  for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
    if a >= scale:
      return f"{sign}${a / scale:.2f}{unit}"
  if a >= 1000:
    return f"{sign}${a:,.0f}"
  return f"{sign}${a:,.2f}"


def pct(v, digits = 1):
  """A fraction as a percent — ``0.18`` → ``18.0%``."""
  if not _is_real(v):
    return "n/a"
  return f"{v * 100:.{digits}f}%"


def num(v, digits = 2):
  """A plain decimal (ratios, betas, multiples) — ``n/a`` for None/NaN."""
  if not _is_real(v):
    return "n/a"
  return f"{v:.{digits}f}"
