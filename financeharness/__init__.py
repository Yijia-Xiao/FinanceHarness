"""FinanceHarness: a model-agnostic finance deep-research agent harness.

Modules:
- providers/  model-agnostic backbone seam (native Gemini + OpenAI-compatible chat)
- runtime/    agent loop, never-raise dispatch, registries, chaining, recovery
- tools/      research (search/visit/cite) + equity/market data + valuation/risk compute
- skills/     workflow recipes loaded on demand
- service/    HTTP+SSE service exposing the harness
"""

__version__ = "0.1.0"
