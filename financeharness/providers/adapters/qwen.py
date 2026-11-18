"""Qwen (open-weight, vLLM-served) quirk adapter.

Qwen models served via vLLM enable "thinking mode" per-request through the chat
template and recommend a ``top_k`` alongside the temperature/top_p defaults.
vLLM accepts both through ``extra_body``. Kept off the vanilla path so a hosted
OpenAI-compatible call never sends these.
"""

from __future__ import annotations

import copy

_DEFAULT_TOP_K = 20


def qwen_adapter(kwargs):
    """Inject ``top_k`` and ``enable_thinking`` into ``extra_body`` without

    clobbering any caller-provided extra_body keys. Pure (returns a new dict).
    """
    out = copy.deepcopy(kwargs)
    extra_body = dict(out.get("extra_body") or {})
    extra_body.setdefault("top_k", _DEFAULT_TOP_K)
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", True)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    out["extra_body"] = extra_body
    return out
