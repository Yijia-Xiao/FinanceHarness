"""Context-budget accounting: pure token estimation + the compaction trigger.

The loop should compact the prompt *before* it overflows, not only react to a
400. This is the pure decision layer (no I/O): estimate a message list's token
footprint via a conservative char heuristic (~4 chars/token + per-message
overhead, rounded up so the trigger fires a little early) and decide whether it
exceeds ``context_window − max_tokens − buffer``.
"""

from __future__ import annotations

import json

_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD_TOKENS = 4


def _message_chars(message):
  total = 0
  content = message.get("content")
  if isinstance(content, str):
    total += len(content)
  elif content is not None:
    total += len(json.dumps(content, ensure_ascii=False))
  tcs = message.get("tool_calls")
  if tcs:
    total += len(json.dumps(tcs, ensure_ascii=False))
  for k in ("name", "tool_call_id"):
    v = message.get(k)
    if isinstance(v, str):
      total += len(v)
  return total


def estimate_tokens(messages):
  """Conservative token estimate (char heuristic + per-message overhead)."""
  if not messages:
    return 0
  chars = sum(_message_chars(m) for m in messages if isinstance(m, dict))
  overhead = _PER_MESSAGE_OVERHEAD_TOKENS * len(messages)
  return (chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN + overhead


def compaction_threshold(
    *, context_window, max_tokens, buffer_tokens
):
  """Largest prompt-token count that still leaves room for ``max_tokens`` of

  output plus ``buffer_tokens`` of safety. Floored at 0.
  """
  return max(0, context_window - max_tokens - buffer_tokens)


def over_budget(
    messages,
    *,
    context_window,
    max_tokens,
    buffer_tokens,
):
  """True when the estimated prompt exceeds the compaction threshold."""
  threshold = compaction_threshold(
      context_window=context_window,
      max_tokens=max_tokens,
      buffer_tokens=buffer_tokens,
  )
  return estimate_tokens(messages) > threshold
