"""Shared dated-price helpers for the risk tools. Pure, no network.

Series are lists of points keyed by date (the ``bars`` of data.equity.prices).
Paired statistics join the series on date, so each return compares the same two
sessions in every series.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from financeharness.runtime.tool_registry import ToolError
from financeharness.tools.compute.numeric import Finite

PriceField = Literal["adj_close", "close"]
# Two series are on different bar intervals when their smallest date spacing
# differs by more than this factor (daily is 1 day, weekly 7, monthly 28 to 31).
# The smallest gap is used because weekends and missing sessions only widen gaps.
_SPACING_TOLERANCE = 1.5

POINTS_DESCRIPTION = (
    "dated points, each with date (YYYY-MM-DD) and adj_close (or close with"
    " price_field='close'); other keys are ignored. Chain a whole series as"
    " prev:<call_id>.bars from data_equity_prices"
)
PRICE_FIELD_DESCRIPTION = (
    "Which price to read from each point. adj_close (default) includes"
    " dividends, which returns need; close is the price as traded."
)


class PricePoint(BaseModel):
  """One dated observation of a price series."""

  model_config = ConfigDict(extra="ignore")

  date: datetime.date
  adj_close: Finite | None = Field(None, gt=0)
  close: Finite | None = Field(None, gt=0)


def unwrap_series(value):
  """Unwrap ``[[point, ...]]`` to ``[point, ...]``.

  A whole-series reference written inside a list (``["prev:c1.bars"]``)
  resolves to a list holding the series.
  """
  if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
    return value[0]
  return value


def dated_prices(points, field, name):
  """Map date to price for one series, in date order."""
  out: dict[datetime.date, float] = {}
  for p in points:
    value = getattr(p, field)
    if value is None:
      raise ToolError(f"{name}: the point dated {p.date} has no {field}")
    if p.date in out:
      raise ToolError(
          f"{name}: more than one point dated {p.date}; pass one point per"
          " date (daily or longer bars)"
      )
    out[p.date] = value
  return dict(sorted(out.items()))


def _spacing_days(dates):
  gaps = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
  return min(gaps) if gaps else None


def join_on_date(series, min_points):
  """Restrict named dated series to their common dates.

  ``series`` maps name to ``{date: price}``. Returns ``(dates, prices,
  dropped)``: the common dates in order, each series' prices on them, and per
  series the number of points that had no partner. Raises ToolError when the
  series are on different bar intervals or share fewer than ``min_points``
  dates.
  """
  spacing = {k: _spacing_days(list(v)) for k, v in series.items()}
  known = {k: s for k, s in spacing.items() if s}
  if known and max(known.values()) > _SPACING_TOLERANCE * min(known.values()):
    listed = ", ".join(f"{k} every {s} day(s)" for k, s in known.items())
    raise ToolError(
        f"series are on different bar intervals ({listed}); fetch them all"
        " with the same interval"
    )
  common = set.intersection(*(set(v) for v in series.values()))
  dates = sorted(common)
  if len(dates) < min_points:
    raise ToolError(
        f"series share {len(dates)} date(s); at least {min_points} common"
        " dates are needed"
    )
  prices = {k: [v[d] for d in dates] for k, v in series.items()}
  dropped = {k: len(v) - len(dates) for k, v in series.items()}
  return dates, prices, dropped


def joint_returns(prices):
  """Period returns per series over the same steps.

  ``prices`` maps name to positive prices on one shared list of dates.
  """
  return {
      k: [b / a - 1.0 for a, b in zip(v, v[1:], strict=False)]
      for k, v in prices.items()
  }
