"""Quirk adapters: per-model transforms of the chat-completion kwargs.

Most backbones need none (the vanilla kwargs work). An adapter exists for the
handful of dialects that differ: ``qwen`` (open-weight models served via vLLM
want ``enable_thinking`` via ``chat_template_kwargs`` and a ``top_k``) and
``openai`` (reasoning models take ``max_completion_tokens`` and reject the
sampling params).

An adapter is ``(kwargs: dict) -> dict`` and must not mutate its input. Selected
by ``ModelProfile.adapter``; ``None`` → identity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from financeharness.providers.adapters.openai import openai_adapter
from financeharness.providers.adapters.qwen import qwen_adapter

Adapter = Callable[[dict[str, Any]], dict[str, Any]]

_ADAPTERS: dict[str, Adapter] = {
    "qwen": qwen_adapter,
    "openai": openai_adapter,
}


def apply_adapter(name, kwargs):
  """Apply the named adapter to ``kwargs``; identity when name is falsy or

  unknown (an unknown name degrades gracefully rather than failing the call).
  """
  if not name:
    return kwargs
  adapter = _ADAPTERS.get(name)
  return adapter(kwargs) if adapter is not None else kwargs


__all__ = ["Adapter", "apply_adapter", "openai_adapter", "qwen_adapter"]
