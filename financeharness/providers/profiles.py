"""Model profiles — the config that selects *which* backbone a call uses.

A profile names a model, an OpenAI-compatible endpoint, the credential to use,
generation defaults, and an optional quirk adapter. Profiles come from built-in
defaults merged with an optional JSON file (``configs/providers.json``); the
active default is chosen by ``FH_PROFILE`` env > file ``"default"`` key >
``"gemini"`` (the built-in cloud default).

Design mirrors ``runtime/config.py`` precedence (defaults < file < env) and
never raises on a missing/malformed file — the seam always has a usable default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Repo-relative default location for the profiles file (optional).
_DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "providers.json"
)


class Generation(BaseModel):
  """Sampling/budget defaults for a profile.

  ``max_tokens`` here is the starting budget; the recovery layer may escalate
  it.
  """

  model_config = ConfigDict(extra="forbid")

  temperature: float = 0.7
  top_p: float = 1.0
  max_tokens: int = 4096
  presence_penalty: float = 0.0


class ModelProfile(BaseModel):
  """A single backbone target.

  ``base_url=None`` uses the OpenAI default endpoint; set it to e.g.
  ``http://127.0.0.1:8000/v1`` for a local vLLM. The API key is resolved at
  client-construction time (never stored).
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  model: str
  base_url: str | None = None
  api_key_env: str = "OPENAI_API_KEY"
  api_key_literal: str | None = None  # e.g. "EMPTY" for an open vLLM
  timeout_s: float = 600.0
  generation: Generation = Field(default_factory=Generation)
  extra_body: dict[str, Any] = Field(default_factory=dict)
  # Extra HTTP headers sent with every Chat Completions call — the header side of
  # the wire, mirroring ``extra_body``. Aggregators want app attribution here
  # (OpenRouter's HTTP-Referer / X-Title); most endpoints need none.
  extra_headers: dict[str, str] = Field(default_factory=dict)
  adapter: str | None = None  # quirk-adapter name (see providers/adapters)
  # The wire protocol the loop uses (providers/base.Provider): "chat" =
  # OpenAI-compatible Chat Completions (works with vLLM / any compatible endpoint);
  # "gemini" = the native google-genai SDK. Selects the provider via provider_for().
  provider: Literal["chat", "gemini"] = "chat"
  # Role in the harness: a "backbone" drives the agent loop (and is switchable via
  # /model); a "reader" is the page-reading model and is never a loop backbone.
  role: Literal["backbone", "reader"] = "backbone"
  # The reader (page-extraction) profile to pair with this backbone. None → the
  # default qwen-reader. Cloud backbones read their own pages so a run never
  # depends on the local reader being served.
  reader_profile: str | None = None

  def resolve_api_key(self, env = None):
    """Resolve the credential: explicit literal wins, else the env var,

    else "" (the client falls back to a placeholder for keyless vLLM).
    """
    if self.api_key_literal is not None:
      return self.api_key_literal
    return (env if env is not None else os.environ).get(self.api_key_env, "")


# Built-in backbones — the runtime that serves the model is the axis of identity:
#   hosted:  gemini (Google) · gpt (OpenAI) · openrouter (aggregator) — need an API key
#   local:   qwen (open-weight, via vLLM)      — keyless, self-served
# The default is `gemini` (a cloud backbone that runs with only an API key, so a
# fresh install works without standing up a server). Each backbone names the
# reader it pairs with; the cloud backbones read their own pages, so a run never
# depends on a local server. Per-profile base_url/model are overridable via
# FH_<NAME>_BASE_URL / FH_<NAME>_MODEL; the qwen `model` ids should match your
# vLLM ``--served-model-name``.
_QWEN_BASE_URL = "http://127.0.0.1:8000/v1"  # local vLLM (open-weight backbone)
_QWEN_READER_BASE_URL = (  # local vLLM (open-weight reader)
    "http://127.0.0.1:8001/v1"
)
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # aggregator (many vendors)

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "gemini": {
        "name": "gemini",
        "model": "gemini-3.6-flash",
        # Native google-genai SDK in the loop — structured thoughts + thought_signature
        # threading (clean multi-turn tool calling). Needs GEMINI_API_KEY. base_url is
        # the OpenAI-compat endpoint, used only by the page-reader's complete() path.
        "provider": "gemini",
        "base_url": _GEMINI_BASE_URL,
        "api_key_env": "GEMINI_API_KEY",
        "reader_profile": "gemini",  # flash reads its own pages (compat path)
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 16384},
    },
    "gpt": {
        "name": "gpt",
        "model": "gpt-5.6",
        # Hosted OpenAI over Chat Completions. base_url=None → the OpenAI default
        # endpoint; needs OPENAI_API_KEY. The `openai` adapter handles reasoning-model
        # kwargs (max_completion_tokens, no sampling params).
        "api_key_env": "OPENAI_API_KEY",
        "adapter": "openai",
        "reader_profile": "gpt",  # reads its own pages (Chat Completions path)
        "generation": {"max_tokens": 16384},
    },
    "openrouter": {
        "name": "openrouter",
        # An aggregator, not a single vendor: one key reaches models from many
        # providers over the same OpenAI-compatible Chat Completions wire, so the
        # vanilla path serves it with no adapter. The `model` is an OpenRouter slug
        # (`vendor/model`) and is the knob you actually turn — override it per run
        # with FH_OPENROUTER_MODEL, or pin one in configs/providers.json. Needs
        # OPENROUTER_API_KEY.
        "model": "anthropic/claude-sonnet-4.5",
        "base_url": _OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        # Optional app attribution on openrouter.ai (rankings + dashboard); harmless
        # to send, and overridable like any other profile field.
        "extra_headers": {
            "HTTP-Referer": "https://github.com/Yijia-Xiao/FinanceHarness",
            "X-Title": "FinanceHarness",
        },
        "reader_profile": "openrouter",  # reads its own pages (Chat Completions path)
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 16384},
    },
    "qwen": {
        "name": "qwen",
        "model": "qwen36-27b-fp8",
        "base_url": _QWEN_BASE_URL,
        "api_key_literal": "EMPTY",
        "adapter": "qwen",  # thinking mode + top_k
        "reader_profile": "qwen-reader",  # the local reader
        "generation": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 16384},
    },
    "qwen-reader": {
        "name": "qwen-reader",
        "model": "reader",
        "role": "reader",
        "base_url": _QWEN_READER_BASE_URL,
        "api_key_literal": "EMPTY",
        # Reader does fast JSON extraction — thinking OFF (top_k via extra_body).
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192},
    },
}

_DEFAULT_PROFILE_NAME = "gemini"


def _read_file(path):
  """Best-effort JSON read; returns {} on any problem (never raises)."""
  try:
    return json.loads(path.read_text())
  except Exception:  # noqa: BLE001 — config is best-effort; fall back to defaults
    return {}


def load_profiles(
    path = None,
    env = None,
):
  """Load all profiles: built-ins merged with the JSON file's ``profiles``

  (file wins per-name). Validation errors on a single file profile are
  skipped, not fatal.
  """
  raw: dict[str, dict[str, Any]] = {
      k: dict(v) for k, v in _BUILTIN_PROFILES.items()
  }

  p = Path(path) if path is not None else _DEFAULT_PROFILES_PATH
  file_data = _read_file(p) if p.exists() else {}
  for name, spec in (file_data.get("profiles") or {}).items():
    merged = {**raw.get(name, {}), **spec, "name": name}
    raw[name] = merged

  e = env if env is not None else os.environ
  profiles: dict[str, ModelProfile] = {}
  for name, spec in raw.items():
    # Per-profile overrides: FH_<NAME>_BASE_URL / FH_<NAME>_MODEL (NAME upper, - → _).
    key = f"FH_{name.upper().replace('-', '_')}"
    base_override = e.get(f"{key}_BASE_URL")
    model_override = e.get(f"{key}_MODEL")
    if base_override:
      spec = {**spec, "base_url": base_override}
    if model_override:
      spec = {**spec, "model": model_override}
    try:
      profiles[name] = ModelProfile(**spec)
    except Exception:  # noqa: BLE001 — skip a malformed profile, keep the rest
      continue
  return profiles


def default_profile_name(
    path = None,
    env = None,
):
  """Active default: ``FH_PROFILE`` env > file ``"default"`` > built-in (gemini).

  Falls back to the built-in default if the configured name is not a known
  profile, so the reported default and the profile actually used never disagree.
  """
  e = env if env is not None else os.environ
  p = Path(path) if path is not None else _DEFAULT_PROFILES_PATH
  file_default = _read_file(p).get("default") if p.exists() else None
  chosen = e.get("FH_PROFILE") or file_default or _DEFAULT_PROFILE_NAME
  if chosen not in load_profiles(path=path, env=env):
    return _DEFAULT_PROFILE_NAME
  return chosen


def get_profile(
    name = None,
    *,
    path = None,
    env = None,
):
  """Resolve a profile by name (or the active default).

  Falls back to the built-in default (``gemini``) if the requested name is
  unknown.
  """
  profiles = load_profiles(path=path, env=env)
  resolved = name or default_profile_name(path=path, env=env)
  return profiles.get(resolved) or profiles[_DEFAULT_PROFILE_NAME]


def available_backbones(
    *,
    path = None,
    env = None,
):
  """Switchable backbone profiles (``role == "backbone"``) with credential

  availability — the page reader is excluded. ``available`` is True for a
  keyless
  vLLM (``api_key_literal`` set) or when the profile's API key resolves. Ordered
  available first (the default floated up), then alphabetically. Backs the
  ``/models`` endpoint's picker; the active profile is a per-request choice, so
  the endpoint reports the configured default separately.
  """
  e = env if env is not None else os.environ
  active = default_profile_name(path=path, env=env)
  out: list[dict[str, Any]] = []
  for p in load_profiles(path=path, env=env).values():
    if (
        p.role != "backbone"
    ):  # readers are paired internally, not user-selectable
      continue
    available = p.api_key_literal is not None or bool(p.resolve_api_key(e))
    out.append({"name": p.name, "model": p.model, "available": available})
  # available first, the default floated to the top, then alphabetical.
  out.sort(key=lambda m: (not m["available"], m["name"] != active, m["name"]))
  return out
