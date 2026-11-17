"""Configuration loading: runtime settings and model profiles, with their precedence rules."""

import json

import pytest

from financeharness.providers.profiles import (
  available_backbones,
  default_profile_name,
  get_profile,
  load_profiles,
)
from financeharness.runtime.config import RuntimeConfig, load_runtime_config


def write_json(path, data):
  path.write_text(json.dumps(data), encoding="utf-8")
  return path


def test_runtime_defaults_apply_without_a_file(tmp_path):
  assert load_runtime_config(tmp_path / "absent.json", env={}) == RuntimeConfig()


def test_runtime_file_overrides_only_the_keys_it_sets(tmp_path):
  path = write_json(
      tmp_path / "runtime.json",
      {
          "_comment": "ignored",
          "limits": {"max_rounds": 7, "_note": "ignored", "not_a_setting": 1},
          "not_a_section": {"x": 1},
      },
  )
  cfg = load_runtime_config(path, env={})
  assert cfg.limits.max_rounds == 7
  assert cfg.limits.run_wall_clock_s == RuntimeConfig().limits.run_wall_clock_s
  assert cfg.generation == RuntimeConfig().generation


def test_runtime_precedence_is_file_then_overrides_then_environment(tmp_path):
  path = write_json(
      tmp_path / "runtime.json",
      {"limits": {"max_rounds": 7, "per_call_timeout_s": 11}, "generation": {"max_tokens": 100}},
  )
  cfg = load_runtime_config(
      path,
      overrides={"limits": {"max_rounds": 8}, "generation": {"max_tokens": 200}},
      env={"MAX_TOKENS": "300", "CONTEXT_WINDOW": "5000", "MAX_RUN_DURATION_S": "60"},
  )
  assert cfg.limits.per_call_timeout_s == 11
  assert cfg.limits.max_rounds == 8
  assert cfg.generation.max_tokens == 300
  assert cfg.recovery.context_window == 5000
  assert cfg.limits.run_wall_clock_s == 60


def test_unparseable_environment_value_keeps_the_lower_layer(tmp_path):
  path = write_json(tmp_path / "runtime.json", {"limits": {"max_rounds": 7}})
  cfg = load_runtime_config(path, env={"MAX_LLM_CALL_PER_RUN": "many", "MAX_TOKENS": ""})
  assert cfg.limits.max_rounds == 7
  assert cfg.generation.max_tokens == RuntimeConfig().generation.max_tokens


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_runtime_file_that_is_not_a_json_object_is_ignored(tmp_path, content):
  path = tmp_path / "runtime.json"
  path.write_text(content, encoding="utf-8")
  assert load_runtime_config(path, env={}) == RuntimeConfig()


def test_runtime_file_with_a_wrong_value_type_falls_back_to_defaults(tmp_path):
  path = write_json(tmp_path / "runtime.json", {"limits": {"max_rounds": "many"}})
  assert load_runtime_config(path, env={}).limits.max_rounds == RuntimeConfig().limits.max_rounds


@pytest.mark.parametrize(
    "content",
    [{"limits": 5}, {"reader": {"max_tokens": [1]}}],
)
def test_runtime_file_that_does_not_fit_the_schema_is_ignored_as_a_whole(tmp_path, content):
  content["generation"] = {"max_tokens": 1234}
  path = write_json(tmp_path / "runtime.json", content)
  assert load_runtime_config(path, env={}) == RuntimeConfig()


def test_builtin_profiles_load_without_a_file(tmp_path):
  profiles = load_profiles(tmp_path / "absent.json", env={})
  assert set(profiles) == {"gemini", "gpt", "qwen", "qwen-reader"}
  assert profiles["gemini"].provider == "gemini"
  assert profiles["qwen-reader"].role == "reader"
  assert default_profile_name(tmp_path / "absent.json", env={}) == "gemini"


def test_profile_file_adds_and_overrides_profiles_and_skips_invalid_ones(tmp_path):
  path = write_json(
      tmp_path / "providers.json",
      {
          "profiles": {
              "gpt": {"model": "other-model"},
              "local": {"model": "served", "base_url": "https://llm.example.com/v1"},
              "broken": {"model": "m", "no_such_field": 1},
          }
      },
  )
  profiles = load_profiles(path, env={})
  assert profiles["gpt"].model == "other-model"
  assert profiles["gpt"].adapter == "openai"
  assert profiles["local"].name == "local"
  assert profiles["local"].base_url == "https://llm.example.com/v1"
  assert "broken" not in profiles
  assert "gemini" in profiles


def test_environment_overrides_model_and_endpoint_per_profile(tmp_path):
  env = {
      "FH_QWEN_READER_MODEL": "small-reader",
      "FH_QWEN_READER_BASE_URL": "https://reader.example.com/v1",
      "FH_GPT_MODEL": "pinned",
  }
  profiles = load_profiles(tmp_path / "absent.json", env=env)
  assert profiles["qwen-reader"].model == "small-reader"
  assert profiles["qwen-reader"].base_url == "https://reader.example.com/v1"
  assert profiles["gpt"].model == "pinned"
  assert profiles["qwen"].model != "small-reader"


@pytest.mark.parametrize(
    ("file_default", "env", "expected"),
    [
        ("gpt", {}, "gpt"),
        ("gpt", {"FH_PROFILE": "qwen"}, "qwen"),
        (None, {"FH_PROFILE": "qwen"}, "qwen"),
        ("gpt", {"FH_PROFILE": "no-such-profile"}, "gemini"),
        ("no-such-profile", {}, "gemini"),
    ],
)
def test_default_profile_is_environment_then_file_then_builtin(
    tmp_path, file_default, env, expected
):
  path = write_json(tmp_path / "providers.json", {"default": file_default} if file_default else {})
  assert default_profile_name(path, env=env) == expected
  assert get_profile(path=path, env=env).name == expected


def test_unknown_profile_name_resolves_to_the_builtin_default(tmp_path):
  assert get_profile("no-such-profile", path=tmp_path / "absent.json", env={}).name == "gemini"


def test_malformed_profile_file_is_ignored(tmp_path):
  path = tmp_path / "providers.json"
  path.write_text("{not json", encoding="utf-8")
  assert set(load_profiles(path, env={})) == {"gemini", "gpt", "qwen", "qwen-reader"}
  assert default_profile_name(path, env={}) == "gemini"


def test_api_key_comes_from_the_literal_or_the_named_variable(tmp_path):
  profiles = load_profiles(tmp_path / "absent.json", env={})
  assert profiles["qwen"].resolve_api_key({"OPENAI_API_KEY": "from-env"}) == "EMPTY"
  assert profiles["gpt"].resolve_api_key({"OPENAI_API_KEY": "from-env"}) == "from-env"
  assert profiles["gpt"].resolve_api_key({}) == ""


def test_backbone_list_excludes_readers_and_puts_usable_ones_first(tmp_path):
  path = tmp_path / "absent.json"
  no_keys = available_backbones(path=path, env={})
  with_key = available_backbones(path=path, env={"GEMINI_API_KEY": "from-env"})
  assert [(m["name"], m["available"]) for m in no_keys] == [
      ("qwen", True),
      ("gemini", False),
      ("gpt", False),
  ]
  assert [(m["name"], m["available"]) for m in with_key] == [
      ("gemini", True),
      ("qwen", True),
      ("gpt", False),
  ]
