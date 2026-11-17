"""Data tools: vendor-agnostic interfaces, yfinance implementation.

Data tools are stateless (unlike the research trio), so they're module-level
``ToolSpec`` constants registered into a registry. Tier is ``deferred``: they're
catalog-only until ``load_tool`` (or a skill) pulls them, keeping the
orchestrator's context lean.
"""
