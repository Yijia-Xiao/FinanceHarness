"""Request assembly for the Chat Completions path."""

import pytest

from financeharness.providers.client import build_create_kwargs
from financeharness.providers.profiles import ModelProfile


def test_output_budget_is_passed_by_the_caller_not_read_from_the_profile():
  profile = ModelProfile(name="p", model="m")
  assert "max_tokens" not in build_create_kwargs(profile, [])
  assert build_create_kwargs(profile, [], max_tokens=512)["max_tokens"] == 512
  with pytest.raises(ValueError, match="max_tokens"):
    ModelProfile(name="p", model="m", generation={"max_tokens": 4096})
