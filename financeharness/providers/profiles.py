"""Model profiles: the config that selects *which* backbone a call uses.

A profile names a model, an OpenAI-compatible endpoint, the credential to use,
generation defaults, and an optional quirk adapter. Profiles come from built-in
defaults merged with the user's ``providers.json`` (see
``financeharness/paths.py``).

Never raises on a missing or malformed file: the built-in profiles are always
available, and every part of the file that is ignored is reported on stderr.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from financeharness.paths import config_file, read_json_object, reason_of, warn_once

PROFILES_FILE = "providers.json"


class Generation(BaseModel):
    """Sampling defaults for a profile.

    The output token budget is not a profile setting: callers pass it from the
    runtime config (``generation.max_tokens``, ``reader.max_tokens``).
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.7
    top_p: float = 1.0
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
    adapter: str | None = None  # quirk-adapter name (see providers/adapters)
    # The wire protocol the loop uses (providers/base.Provider): "chat" =
    # OpenAI-compatible Chat Completions (works with vLLM / any compatible endpoint);
    # "gemini" = the native google-genai SDK. Selects the provider via provider_for().
    provider: Literal["chat", "gemini"] = "chat"
    # Role in the harness: a "backbone" drives the agent loop (and is selectable per
    # request); a "reader" is the page-reading model and is never a loop backbone.
    role: Literal["backbone", "reader"] = "backbone"
    # The reader (page-extraction) profile to pair with this backbone. None → the
    # default qwen-reader. Cloud backbones read their own pages so a run never
    # depends on the local reader being served.
    reader_profile: str | None = None
    # Context window in tokens. Bounds the page text sent when this profile reads
    # pages; None falls back to the runtime ``recovery.context_window``.
    context_window: int | None = None

    def resolve_api_key(self, env=None):
        """Resolve the credential: explicit literal wins, else the env var,

        else "" (the client falls back to a placeholder for keyless vLLM).
        """
        if self.api_key_literal is not None:
            return self.api_key_literal
        return (env if env is not None else os.environ).get(self.api_key_env, "")


# Built-in backbones. The runtime that serves the model is the axis of identity:
#   hosted (need an API key):     gemini (Google), gpt (OpenAI)
#   local (keyless, self-served): qwen (open-weight, via vLLM)
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

_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "gemini": {
        "name": "gemini",
        "model": "gemini-3.6-flash",
        # Native google-genai SDK in the loop: structured thoughts + thought_signature
        # threading (clean multi-turn tool calling). Needs GEMINI_API_KEY. base_url is
        # the OpenAI-compat endpoint, used only by the page-reader's complete() path.
        "provider": "gemini",
        "base_url": _GEMINI_BASE_URL,
        "api_key_env": "GEMINI_API_KEY",
        "reader_profile": "gemini",  # flash reads its own pages (compat path)
        "context_window": 1048576,
        "generation": {"temperature": 0.7, "top_p": 0.95},
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
    },
    "qwen": {
        "name": "qwen",
        "model": "qwen36-27b-fp8",
        "base_url": _QWEN_BASE_URL,
        "api_key_literal": "EMPTY",
        "adapter": "qwen",  # thinking mode + top_k
        "reader_profile": "qwen-reader",  # the local reader
        "generation": {"temperature": 0.6, "top_p": 0.95},
    },
    "qwen-reader": {
        "name": "qwen-reader",
        "model": "reader",
        "role": "reader",
        "base_url": _QWEN_READER_BASE_URL,
        "api_key_literal": "EMPTY",
        # Reader does fast JSON extraction: thinking OFF (top_k via extra_body).
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "generation": {"temperature": 0.7, "top_p": 0.95},
    },
}

_DEFAULT_PROFILE_NAME = "gemini"


def _read_file(path):
    """The profiles file as a dict: empty when absent or, with a warning, unusable."""
    try:
        return read_json_object(path) or {}
    except (OSError, ValueError) as err:
        warn_once(f"ignoring {path}: {reason_of(err)}; using the built-in profiles")
        return {}


def _with_env(name, spec, env):
    """``spec`` with the ``FH_<NAME>_BASE_URL`` and ``FH_<NAME>_MODEL`` overrides."""
    key = f"FH_{name.upper().replace('-', '_')}"
    base_override = env.get(f"{key}_BASE_URL")
    model_override = env.get(f"{key}_MODEL")
    if base_override:
        spec = {**spec, "base_url": base_override}
    if model_override:
        spec = {**spec, "model": model_override}
    return spec


def load_profiles(
    path=None,
    env=None,
):
    """Load all profiles: the built-ins, changed or extended by the file's ``profiles``.

    A file profile that is not valid is reported and skipped; when it overrides
    a built-in profile, the built-in one is kept unchanged.
    """
    e = env if env is not None else os.environ
    p = Path(path) if path is not None else config_file(PROFILES_FILE, e)
    file_profiles = _read_file(p).get("profiles") or {}
    if not isinstance(file_profiles, dict):
        warn_once(f'ignoring "profiles" in {p}: it is not an object')
        file_profiles = {}

    profiles = {
        name: ModelProfile(**_with_env(name, spec, e)) for name, spec in _BUILTIN_PROFILES.items()
    }
    for name, spec in file_profiles.items():
        try:
            if not isinstance(spec, dict):
                raise ValueError("it is not an object")
            merged = {**_BUILTIN_PROFILES.get(name, {}), **spec, "name": name}
            profiles[name] = ModelProfile(**_with_env(name, merged, e))
        except (ValueError, TypeError) as err:
            kept = "; keeping the built-in profile" if name in _BUILTIN_PROFILES else ""
            warn_once(f"ignoring profile '{name}' in {p}: {reason_of(err)}{kept}")
    return profiles


def default_profile_name(
    path=None,
    env=None,
):
    """The active default profile: ``FH_PROFILE``, else the file's ``"default"``.

    A name that is not a known profile is reported and the built-in default is
    used, so the reported default and the profile actually used never disagree.
    """
    e = env if env is not None else os.environ
    p = Path(path) if path is not None else config_file(PROFILES_FILE, e)
    known = load_profiles(path=path, env=env)
    for source, chosen in (
        ("FH_PROFILE", e.get("FH_PROFILE")),
        (f'"default" in {p}', _read_file(p).get("default")),
    ):
        if not chosen:
            continue
        if isinstance(chosen, str) and chosen in known:
            return chosen
        warn_once(
            f"ignoring {source}: {chosen!r} is not a known profile; using '{_DEFAULT_PROFILE_NAME}'"
        )
        break
    return _DEFAULT_PROFILE_NAME


def get_profile(
    name=None,
    *,
    path=None,
    env=None,
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
    path=None,
    env=None,
):
    """Switchable backbone profiles (``role == "backbone"``) with credential availability.

    The page reader is excluded. ``available`` is True for a keyless vLLM
    (``api_key_literal`` set) or when the profile's API key resolves. Ordered
    available first (the default floated up), then alphabetically. Backs the
    ``/models`` endpoint's picker; the active profile is a per-request choice, so
    the endpoint reports the configured default separately.
    """
    e = env if env is not None else os.environ
    active = default_profile_name(path=path, env=env)
    out: list[dict[str, Any]] = []
    for p in load_profiles(path=path, env=env).values():
        if p.role != "backbone":  # readers are paired internally, not user-selectable
            continue
        available = p.api_key_literal is not None or bool(p.resolve_api_key(e))
        out.append({"name": p.name, "model": p.model, "available": available})
    # available first, the default floated to the top, then alphabetical.
    out.sort(key=lambda m: (not m["available"], m["name"] != active, m["name"]))
    return out
