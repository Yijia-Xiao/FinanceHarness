#!/usr/bin/env python3
"""A local MCP data source: your own portfolio, private to this machine.

The point of the inbound MCP path is that the harness can reason over data it has
no business copying. This server reads a CSV that never leaves the box, exposes it
as three tools and one resource, and speaks MCP over stdin/stdout — so
FinanceHarness (or Claude Desktop, or any MCP client) can ask about your holdings
while the file stays where it is.

Run it directly to serve on stdio:

    uv run python examples/mcp/local_portfolio.py

Or let the harness start it for you — enable the ``portfolio`` entry in
``configs/mcp.json`` and every run gets these tools in its catalog:

    fh -p "Given my holdings, what is my concentration risk in semis?"

Point it at your own file with ``FH_PORTFOLIO_CSV=/path/to/holdings.csv``.
Columns: ticker, shares, cost_basis, opened.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.resources.types import TextResource

DEFAULT_CSV = Path(__file__).with_name("holdings.csv")

server = MCPServer(
    name="portfolio",
    title="Local portfolio",
    version="0.1.0",
    instructions=(
        "A private, local holdings file. Use portfolio_holdings for the whole book,"
        " portfolio_position for one ticker, and portfolio_tickers when you only"
        " need the symbols to research. Cost basis is per share."
    ),
)


def _csv_path():
  """The holdings file: ``FH_PORTFOLIO_CSV`` if set, else the bundled sample."""
  return Path(os.environ.get("FH_PORTFOLIO_CSV") or DEFAULT_CSV)


def _rows():
  """Parsed holdings, newest column set wins. Raises if the file is missing."""
  path = _csv_path()
  if not path.is_file():
    raise FileNotFoundError(f"holdings file not found: {path}")
  with path.open(newline="") as fh:
    return [
        {
            "ticker": (row.get("ticker") or "").strip().upper(),
            "shares": float(row.get("shares") or 0),
            "cost_basis": float(row.get("cost_basis") or 0),
            "opened": (row.get("opened") or "").strip(),
        }
        for row in csv.DictReader(fh)
        if (row.get("ticker") or "").strip()
    ]


def _table(rows):
  head = "| Ticker | Shares | Cost basis | Cost value | Opened |"
  sep = "| --- | ---: | ---: | ---: | --- |"
  body = [
      f"| {r['ticker']} | {r['shares']:,.0f} | {r['cost_basis']:,.2f} |"
      f" {r['shares'] * r['cost_basis']:,.2f} | {r['opened']} |"
      for r in rows
  ]
  total = sum(r["shares"] * r["cost_basis"] for r in rows)
  return "\n".join([head, sep, *body]) + f"\n\nTotal cost basis: {total:,.2f}"


@server.tool(name="portfolio_holdings")
def portfolio_holdings() -> str:
  """Every position in the local portfolio: ticker, share count, per-share cost
  basis, cost value, and the date the position was opened. Read this before
  answering anything about "my portfolio" or "my holdings"."""
  rows = _rows()
  if not rows:
    return "The holdings file is empty."
  return f"{len(rows)} positions in {_csv_path().name}:\n\n{_table(rows)}"


@server.tool(name="portfolio_position")
def portfolio_position(ticker: str) -> str:
  """One position by ticker — share count, per-share cost basis, cost value and
  open date. Returns a clear miss when the ticker is not held, which is itself
  the answer to "do I own X?"."""
  wanted = ticker.strip().upper()
  match = [r for r in _rows() if r["ticker"] == wanted]
  if not match:
    return f"{wanted} is not held in {_csv_path().name}."
  return _table(match)


@server.tool(name="portfolio_tickers")
def portfolio_tickers() -> str:
  """Just the tickers held, comma-separated — the cheap call when you only need
  the symbols to feed into market-data or valuation tools."""
  tickers = [r["ticker"] for r in _rows()]
  return ", ".join(tickers) if tickers else "(no holdings)"


def _register_resource():
  """The raw CSV as a readable resource, for a client that wants the file itself."""
  path = _csv_path()
  try:
    text = path.read_text()
  except OSError:
    return
  server.add_resource(
      TextResource(
          uri="portfolio://holdings.csv",
          name="holdings.csv",
          description=f"The raw local holdings file ({path}).",
          mime_type="text/csv",
          text=text,
      )
  )


if __name__ == "__main__":
  _register_resource()
  server.run("stdio")
