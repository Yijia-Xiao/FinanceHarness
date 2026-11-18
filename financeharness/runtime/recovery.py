"""Model-call recovery policy: pure decisions, no I/O.

Decides what to do when a model call doesn't cleanly succeed:
  - truncation (``finish_reason == "length"``) → escalate ``max_tokens``
  - transient (429/5xx/timeout/connection) → exponential backoff + retry
  - context overflow (400 + overflow signature) → compact the prompt + retry
  - malformed generation (other 400) → regenerate (re-roll, temperature > 0)
  - fatal (401/403/404/bug) → stop, fail fast

Pure functions of their inputs (exception / budget / messages), so the policy
is deterministic without a live server. Numeric knobs come from config.
"""

from __future__ import annotations

import os
import random
import re

from financeharness.runtime.retry import is_default_retryable, status_of

BACKOFF = "backoff"
COMPACT = "compact"
REGENERATE = "regenerate"
FATAL = "fatal"

_CONTEXT_OVERFLOW_SIGNATURES = (
    "maximum context length",
    "reduce the length",
    "longer than the maximum",
    "context length",
)

_ELISION_MARKER = "[elided to fit context window"

# Transient SDK exception class names that carry no HTTP status (the request
# never got a response) and aren't builtin Timeout/Connection subclasses.
_TRANSIENT_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectError",
        "RemoteProtocolError",
    }
)


def _is_transient_by_name(err):
    return any(t.__name__ in _TRANSIENT_EXC_NAMES for t in type(err).__mro__)


def _message_of(err):
    msg = getattr(err, "message", None)
    if not isinstance(msg, str) or not msg:
        msg = str(err)
    return msg


def is_context_overflow(err):
    """True when ``err`` is a prompt-too-long error (HTTP 400 + overflow text)."""
    if status_of(err) != 400:
        return False
    text = _message_of(err).lower()
    return any(sig in text for sig in _CONTEXT_OVERFLOW_SIGNATURES)


class ModelCallError(Exception):
    """Raised when a model call cannot be completed. ``str()`` is the cause, for the user."""


_MAX_CAUSE_CHARS = 300
# Text that looks like an API key: the common vendor prefixes and a long tail.
_KEY_LIKE_RE = re.compile(r"\b(sk-|AIza)[A-Za-z0-9_-]{8,}")
_BAD_KEY_SIGNATURES = ("api key not valid", "api_key_invalid", "invalid api key")


def missing_api_key(profile, env=None):
    """The environment variable to set when ``profile`` cannot run without a key.

    None when a key resolves, or when the endpoint may not need one: a
    self-served OpenAI-compatible endpoint (a ``base_url`` is set) runs keyless,
    and the Gemini SDK also accepts ``GOOGLE_API_KEY``.
    """
    env = os.environ if env is None else env
    if profile.resolve_api_key(env):
        return None
    if profile.provider == "gemini":
        return None if env.get("GOOGLE_API_KEY") else profile.api_key_env
    return profile.api_key_env if profile.base_url is None else None


def _is_bad_key(err):
    """True for an HTTP 400 that reports an invalid API key (Gemini does this)."""
    text = _message_of(err).lower()
    return status_of(err) == 400 and any(sig in text for sig in _BAD_KEY_SIGNATURES)


def _detail_of(err, profile):
    """The error's own message on one line, shortened, with key-like text masked."""
    text = " ".join(_message_of(err).split())
    key = profile.resolve_api_key()
    if len(key) >= 8:
        text = text.replace(key, "<api key>")
    text = _KEY_LIKE_RE.sub(lambda m: f"{m.group(1)}<masked>", text)
    if not text:
        text = "the call timed out" if isinstance(err, TimeoutError) else "no details given"
    return text[:_MAX_CAUSE_CHARS]


def describe_model_error(err, profile, retries=0):
    """One line that tells the user why a model call failed and what to check."""
    status = status_of(err)
    detail = _detail_of(err, profile)
    where = f"profile '{profile.name}', model '{profile.model}'"
    if status in (401, 403) or _is_bad_key(err):
        hint = (
            "check the endpoint's access settings"
            if profile.api_key_literal is not None
            else f"check {profile.api_key_env}"
        )
        cause = f"the API rejected the credentials (HTTP {status}, {where}): {hint}. {detail}"
    elif status == 404:
        cause = (
            f"the model or endpoint was not found (HTTP 404, {where}): check the"
            f" model name and base URL. {detail}"
        )
    else:
        kind = type(err).__name__ + (f", HTTP {status}" if status else "")
        cause = f"the model call failed ({kind}, {where}): {detail}"
    if retries:
        cause += f" (gave up after {retries} {'retry' if retries == 1 else 'retries'})"
    return cause


def classify_model_error(err):
    """Map an exception to COMPACT / REGENERATE / BACKOFF / FATAL.

    The 400 split is by kind, not by matching tool-parser strings: a fixed,
    schema-valid request only yields 400s for (a) context overflow → COMPACT or
    (b) a malformed generation the parser rejected → REGENERATE (clears on a
    re-roll). Anything else transient → BACKOFF; otherwise FATAL.
    """
    if is_context_overflow(err):
        return COMPACT
    if _is_bad_key(err):
        return FATAL
    if status_of(err) == 400:
        return REGENERATE
    if is_default_retryable(err) or _is_transient_by_name(err):
        return BACKOFF
    return FATAL


def escalate_max_tokens(current, ceiling, factor=2):
    """Next ``max_tokens`` after a truncation, or ``None`` if already at the

    ceiling (accept the best-effort partial).
    """
    if current >= ceiling:
        return None
    return min(current * max(factor, 2), ceiling)


def backoff_delay(
    attempt,
    *,
    base_delay_s,
    max_delay_s,
    jitter=True,
    rng=None,
):
    """Exponential backoff with proportional jitter, clamped to ``max_delay_s``.

    Deterministic when ``jitter=False``.
    """
    delay = min(base_delay_s * (2**attempt), max_delay_s)
    if jitter:
        r = (rng or random).uniform(0, delay * 0.25)
        delay = min(delay + r, max_delay_s)
    return delay


def compact_messages(
    messages,
    *,
    keep_recent=4,
):
    """Shrink the prompt by eliding old tool-result bodies (keep the most recent

    ``keep_recent`` intact). Preserves role + tool_call_id linkage; the full
    output stays in the trajectory's tool_log. Returns ``(new_messages,
    n_elided)`` or ``None`` when nothing is left to compact (idempotent). Never
    mutates the input.
    """
    tool_idxs = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    ]
    protected = set(tool_idxs[-keep_recent:]) if keep_recent > 0 else set()
    candidates = [i for i in tool_idxs if i not in protected]

    new_messages = [dict(m) for m in messages]
    n_elided = 0
    for i in candidates:
        content = new_messages[i].get("content") or ""
        if _ELISION_MARKER in content or len(content) <= len(_ELISION_MARKER) + 64:
            continue
        new_messages[i]["content"] = (
            f"{_ELISION_MARKER}: {len(content)} chars omitted; the full tool "
            "output is preserved in the run trajectory's tool_log.]"
        )
        n_elided += 1

    if n_elided == 0:
        return None
    return new_messages, n_elided
