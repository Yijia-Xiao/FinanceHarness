---
name: options-pricing
description: Price listed options on a stock or ETF and read what the market implies — pull the option chain, a risk-free rate and realized volatility, compute Black-Scholes model prices, Greeks and implied volatility, and compare market vs. model prices and implied vs. historical volatility.
tags: [options, derivatives, black-scholes, implied-volatility, greeks]
requires_tools:
  - data_market_rates
  - data_options_chain
  - data_equity_prices
  - data_equity_ratios
  - compute_options_black_scholes
  - compute_options_implied_vol
---

# Options pricing

Set the market's option prices against a Black-Scholes-Merton model and read
what the gap says about expected volatility.

## Workflow

1.  **Chain and spot.** `data_options_chain` for the underlying — it returns the
    listed expirations, the spot, and one expiry's calls and puts near the
    money with `time_to_expiry_years`. Pick the expiry the question is about
    (or the default first expiry a week or more out); call again with
    `expiration` for another tenor.
2.  **Risk-free rate.** `data_market_rates`. For options under about a year,
    use the 3-month Treasury yield (`tenors_pct.3m`, from ^IRX) rather than the
    10-year; convert the percent to a decimal with `calc` (4.3 → 0.043).
3.  **Dividend yield.** `data_equity_ratios` → `ratios.dividend_yield` (a
    decimal); use 0 for a non-payer or when it is unavailable, and say so.
4.  **Historical volatility.** `data_equity_prices` → `realized_vol.annualized`
    (daily log returns, ×√252). Match the window roughly to the option's tenor —
    `period="3mo"` for near-dated options, `"1y"` for longer ones — and report
    which window you used.
5.  **Implied volatility.** `compute_options_implied_vol` for calls and for
    puts, chaining strikes and bid/ask mids from the chain
    (`prev:<id>.calls[*].strike`, `prev:<id>.calls[*].mid`) with the chain's
    spot and `time_to_expiry_years`, the rate, and the dividend yield. Prefer
    mids over last prices: a last trade can be hours old. Where a strike has
    no live bid/ask (`mid` is null — common near the open, after the close,
    and on illiquid strikes), fall back to `last` only with its `last_trade`
    time stated.
6.  **Model prices and Greeks.** `compute_options_black_scholes` at the
    historical volatility for the same strikes — the price the stock's own
    recent movement would justify — with delta, gamma, theta, and vega.
7.  **Context.** `search` + `visit` for what could explain a gap between
    implied and historical volatility before the expiry: an earnings date,
    a product or regulatory event, a macro release. Cite what you read.
8.  **Report.** Market mid vs. model price per strike, implied vs. historical
    volatility (at the money and across strikes, i.e. the smile or skew), the
    Greeks that matter for the question, and the events that plausibly
    account for the difference.

## Principles

-   Yahoo option data is delayed, has no Greeks, and its own implied vol is
    unreliable when there is no live bid/ask (outside market hours, illiquid
    strikes). State the data's timestamp, recompute IV from mids, and flag wide
    spreads, zero bids, and thin open interest rather than reading meaning into
    them.
-   Black-Scholes-Merton prices European exercise at a single flat volatility.
    US equity options are American, so deep in-the-money puts and calls before
    an ex-dividend date can trade above the model; say so where it matters.
-   Implied volatility is the market's expectation, not a forecast you made;
    present implied-minus-historical as a premium (or discount) the market is
    paying, and tie it to a cited catalyst where one exists.
-   Run every number through the tools (chain → rate → vol → price/IV) and
    state the inputs — spot, tenor, rate, dividend yield, volatility window —
    so the comparison can be reproduced.
