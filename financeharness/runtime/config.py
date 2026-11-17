"""Runtime configuration.

One source of truth for the generation budget, run limits, error-recovery
policy, and reader params.

The defaults are the field defaults of the models below. The user's
``runtime.json`` (see ``financeharness/paths.py``), programmatic ``overrides``
and environment variables are layered over them in ``load_runtime_config``. The
loader never raises on a missing or malformed file: the harness always has a
usable config.
Resource limits are anti-runaway *backstops* sized generous; the bounded round
count is what guarantees termination.
"""

from __future__ import annotations

import contextlib
import copy
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from financeharness.paths import config_file, read_json_object, reason_of, warn_once

RUNTIME_FILE = "runtime.json"


class _Section(BaseModel):
  # Tolerate unknown keys in the file rather than crashing the loader.
  model_config = ConfigDict(extra="ignore")


class GenerationConfig(_Section):
  """Main model generation token budget and truncation escalation settings."""

  max_tokens: int = 16384
  max_tokens_ceiling: int = 65536
  max_tokens_escalation_factor: int = 2


class LimitsConfig(_Section):
  """Run, call, round, and retry caps that prevent runaway execution."""

  per_call_timeout_s: int = 1800
  run_wall_clock_s: int = 3600
  max_rounds: int = 100
  max_transient_retries: int = 10
  max_truncation_escalations: int = 3
  max_regenerate_retries: int = 3
  # Blank replies (no text, no tool calls) answered with a nudge before the
  # run ends as "empty_answer".
  max_empty_reply_retries: int = 2


class BackoffConfig(_Section):
  """Transient-error backoff parameters."""

  base_delay_s: float = 1.0
  max_delay_s: float = 30.0
  jitter: bool = True


class RecoveryConfig(_Section):
  """Context-window and compaction controls used by model-call recovery."""

  compaction_enabled: bool = True
  compaction_keep_recent_tool_results: int = 4
  context_window: int = 131072
  compaction_buffer_tokens: int = 4096
  proactive_compaction: bool = True


class ReaderConfig(_Section):
  """Generation budget for the page-reader extraction call, separate from the

  agent's main budget so it's tunable independently.
  """

  max_tokens: int = (
      8192  # generous so reasoning-model readers leave room for output
  )
  # Characters of page text the reader sees. None derives the limit from the
  # reader's context window (visit_reader.page_char_limit).
  max_chars: int | None = Field(None, gt=0)


class FetchConfig(_Section):
  """Page fetching."""

  # User-Agent sent to sec.gov, which refuses browser agents. The SEC asks
  # automated clients for a name and a contact email.
  sec_user_agent: str = "FinanceHarness research agent"

  @field_validator("sec_user_agent")
  @classmethod
  def _header_value(cls, v):
    v = v.strip()
    if not v or not v.isascii() or not v.isprintable():
      raise ValueError("must be non-empty printable ASCII")
    return v


class RuntimeConfig(_Section):
  """All runtime configuration sections after defaults/file/env merging."""

  generation: GenerationConfig = GenerationConfig()
  limits: LimitsConfig = LimitsConfig()
  backoff: BackoffConfig = BackoffConfig()
  recovery: RecoveryConfig = RecoveryConfig()
  reader: ReaderConfig = ReaderConfig()
  fetch: FetchConfig = FetchConfig()


def _sec_user_agent(raw):
  return FetchConfig(sec_user_agent=raw).sec_user_agent


# env var -> (section, key, caster). Highest precedence so a live deployment is
# tunable without editing the file.
_ENV_OVERRIDES: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "MAX_TOKENS": ("generation", "max_tokens", int),
    "PER_CALL_TIMEOUT_S": ("limits", "per_call_timeout_s", int),
    "MAX_RUN_DURATION_S": ("limits", "run_wall_clock_s", int),
    "MAX_LLM_CALL_PER_RUN": ("limits", "max_rounds", int),
    "CONTEXT_WINDOW": ("recovery", "context_window", int),
    "FH_SEC_USER_AGENT": ("fetch", "sec_user_agent", _sec_user_agent),
}


def _strip_comments(value):
  """Drop ``_``-prefixed keys recursively so the file can self-document."""
  if isinstance(value, dict):
    return {
        k: _strip_comments(v)
        for k, v in value.items()
        if not (isinstance(k, str) and k.startswith("_"))
    }
  return value


def _deep_merge(
    base, over
):
  """Recursively merge ``over`` onto a copy of ``base`` (dicts merge; scalars

  replace).
  """
  out = copy.deepcopy(base)
  for k, v in over.items():
    if isinstance(v, Mapping) and isinstance(out.get(k), dict):
      out[k] = _deep_merge(out[k], v)
    else:
      out[k] = v
  return out


def load_runtime_config(
    path = None,
    *,
    env = None,
    overrides = None,
):
  """Load the runtime config from the defaults, the file, ``overrides`` and ``env``.

  ``path`` defaults to ``runtime.json`` in the user config directory. A file
  that cannot be used is ignored as a whole, with one warning on stderr.
  """
  env = os.environ if env is None else env
  merged = RuntimeConfig().model_dump()

  p = Path(path) if path is not None else config_file(RUNTIME_FILE, env)
  try:
    file_cfg = read_json_object(p)
    if file_cfg is not None:
      with_file = _deep_merge(merged, _strip_comments(file_cfg))
      RuntimeConfig(**with_file)  # a wrong value type raises ValueError
      merged = with_file
  except (OSError, ValueError, TypeError) as err:
    warn_once(f"ignoring {p}: {reason_of(err)}; using defaults")

  if overrides:
    merged = _deep_merge(merged, _strip_comments(overrides))

  for env_name, (section, key, cast) in _ENV_OVERRIDES.items():
    raw = env.get(env_name)
    if not raw:
      continue
    # ignore an unparseable env value; keep the lower layer
    with contextlib.suppress(ValueError, TypeError):
      merged[section][key] = cast(raw)

  return RuntimeConfig(**merged)
