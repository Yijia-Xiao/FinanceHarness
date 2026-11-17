"""Options-data tools (yfinance): listed option chains for options-pricing research."""

from financeharness.tools.data.options import chain

OPTIONS_DATA_SPECS = [chain.SPEC]

__all__ = ["OPTIONS_DATA_SPECS", "chain"]
