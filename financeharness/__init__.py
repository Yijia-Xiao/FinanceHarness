"""FinanceHarness — a model-agnostic finance deep-research agent harness.

Modules:
- providers/  model-agnostic backbone seam (native Gemini + OpenAI-compatible chat)
- runtime/    agent loop, never-raise dispatch, registries, chaining, recovery
- tools/      research (search/visit/cite) + equity/market data + valuation/risk compute
- skills/     workflow recipes loaded on demand
- mcp/        MCP in both directions: the harness as an MCP server, and
              external MCP servers borrowed as harness tools
- service/    HTTP+SSE service exposing the harness (and the React web UI)
"""

__version__ = "0.1.0"
