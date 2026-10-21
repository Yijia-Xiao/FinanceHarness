"""Options-compute tools — pure-math European option pricing and implied volatility."""

from financeharness.tools.compute.options import black_scholes, implied_vol

OPTIONS_SPECS = [black_scholes.SPEC, implied_vol.SPEC]

__all__ = ["OPTIONS_SPECS", "black_scholes", "implied_vol"]
