"""compute.risk.correlation: return-correlation matrix across dated price series.

Pure-math. Joins the named series on date, converts the common dates to
returns, and returns the pairwise Pearson correlation matrix. Use to see how
names co-move (diversification, crowding, pair risk).
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field, field_validator, model_validator

from financeharness.runtime.tool_registry import ToolResponse, ToolSpec
from financeharness.tools.compute.risk.returns import (
  POINTS_DESCRIPTION,
  PRICE_FIELD_DESCRIPTION,
  PriceField,
  PricePoint,
  dated_prices,
  join_on_date,
  joint_returns,
  unwrap_series,
)
from financeharness.tools.format import num

_DESCRIPTION = (
    "Return-correlation matrix across two or more named dated price series"
    " (series maps each name to prev:<call_id>.bars from data_equity_prices,"
    " all fetched with the same period and interval). Joins the series on"
    " date and refuses series on different intervals. Returns the pairwise"
    " Pearson correlations of their returns over the common dates. Use for"
    " co-movement, diversification, and pair risk."
)
_MIN_POINTS = 3


class CorrelationRequest(BaseModel):
  """Input for a correlation matrix over named price series."""

  series: dict[str, list[PricePoint]] = Field(
      Ellipsis,
      min_length=2,
      description=(
          f"Map of name to series (2 or more names). Each series: {POINTS_DESCRIPTION},"
          ' e.g. {"AAPL": "prev:c1.bars", "MSFT": "prev:c2.bars"}.'
      ),
  )
  price_field: PriceField = Field("adj_close", description=PRICE_FIELD_DESCRIPTION)

  @field_validator("series", mode="before")
  @classmethod
  def _unwrap(cls, value):
    if isinstance(value, dict):
      return {k: unwrap_series(v) for k, v in value.items()}
    return value

  @model_validator(mode="after")
  def _check_lengths(self):
    short = [k for k, v in self.series.items() if len(v) < _MIN_POINTS]
    if short:
      raise ValueError(
          f"series need >= {_MIN_POINTS} points; too short: {short}"
      )
    return self


def compute_correlation(series, price_field = "adj_close"):
  """Compute pairwise correlations of returns over the dates all series share."""
  names = list(series)
  dates, prices, dropped = join_on_date(
      {k: dated_prices(v, price_field, k) for k, v in series.items()},
      _MIN_POINTS,
  )
  rets = joint_returns(prices)
  n_obs = len(rets[names[0]])
  variances = (
      {k: statistics.variance(v) for k, v in rets.items()} if n_obs >= 2 else {}
  )
  matrix: dict[str, dict[str, float | None]] = {a: {} for a in names}
  for i, a in enumerate(names):
    for b in names[i:]:
      # undefined when either series is constant: report None
      computable = n_obs >= 2 and variances[a] > 0 and variances[b] > 0
      corr = statistics.correlation(rets[a], rets[b]) if computable else None
      c = round(corr, 4) if corr is not None else None
      matrix[a][b] = c
      matrix[b][a] = c
  return {
      "names": names,
      "matrix": matrix,
      "n_obs": n_obs,
      "start": dates[0].isoformat(),
      "end": dates[-1].isoformat(),
      "dropped": dropped,
  }


def _markdown(res):
  names = res["names"]
  head = "corr | " + " | ".join(names)
  rows = [head, "-" * len(head)]
  for a in names:
    rows.append(
        f"{a} | " + " | ".join(num(res["matrix"][a].get(b)) for b in names)
    )
  dropped = ", ".join(f"{k} {v}" for k, v in res["dropped"].items())
  return (
      f"**Return correlation** (n={res['n_obs']}, {res['start']} to"
      f" {res['end']}):\n"
      + "\n".join(rows)
      + f"\n  points without a common date: {dropped}"
  )


async def _handler(req):
  res = compute_correlation(req.series, req.price_field)
  return ToolResponse(markdown=_markdown(res), structured=res, meta={})


SPEC = ToolSpec(
    name="compute_risk_correlation",
    display_name="compute.risk.correlation",
    tier="deferred",
    description=_DESCRIPTION,
    request_schema=CorrelationRequest,
    handler=_handler,
    tags=("compute", "risk", "correlation"),
)
