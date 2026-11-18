"""Tools: research (search/visit/cite) + a minimal equity slice.

Tools that share per-run state (the research trio share a FetchCache) are built
per request via factory functions that close over that state, then registered
into a fresh registry, keeping the per-request-isolation invariant.
"""
