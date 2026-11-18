"""OpenAI reasoning-model quirk adapter.

OpenAI's reasoning models (o-series, GPT-5.x) budget output with
``max_completion_tokens`` instead of ``max_tokens`` and reject the sampling
params (``temperature`` / ``top_p`` / ``presence_penalty``). This renames the
token budget and drops those params. Newer non-reasoning chat models also accept
``max_completion_tokens`` and tolerate the defaults being absent, so this stays
safe as the OpenAI default adapter.
"""

from __future__ import annotations

_UNSUPPORTED_SAMPLING = ("temperature", "top_p", "presence_penalty")


def openai_adapter(kwargs):
    """Rename ``max_tokens`` → ``max_completion_tokens`` and drop the sampling

    params reasoning models reject. Pure (returns a new dict).
    """
    out = dict(kwargs)
    if "max_tokens" in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    for key in _UNSUPPORTED_SAMPLING:
        out.pop(key, None)
    return out
