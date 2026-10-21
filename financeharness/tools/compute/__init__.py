"""Compute tools — pure-math, vendor-agnostic valuation.

No network: compute tools consume numbers (often chained from data tools via
`prev:<id>.<path>`) and return derived measures. Stateless deferred SPECs.
"""

from financeharness.tools.compute.options import OPTIONS_SPECS
from financeharness.tools.compute.risk import RISK_SPECS
from financeharness.tools.compute.valuation import VALUATION_SPECS

COMPUTE_SPECS = [*VALUATION_SPECS, *RISK_SPECS, *OPTIONS_SPECS]

__all__ = ["COMPUTE_SPECS", "OPTIONS_SPECS", "RISK_SPECS", "VALUATION_SPECS"]
