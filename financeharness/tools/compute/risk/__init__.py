"""Risk-compute tools: pure-math risk lenses over price/return series."""

from financeharness.tools.compute.risk import beta, correlation, var

RISK_SPECS = [correlation.SPEC, var.SPEC, beta.SPEC]

__all__ = ["RISK_SPECS", "beta", "correlation", "var"]
