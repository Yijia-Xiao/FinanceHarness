"""Skill registry: SKILL.md parsing, discovery with overrides, and load_skill."""

import pytest

from financeharness.runtime.dispatch import dispatch
from financeharness.runtime.frontmatter import FrontmatterError
from financeharness.runtime.skill_registry import (
  SkillRegistry,
  SkillSessionState,
  load_skill_registry,
  merge_skills,
  parse_skill_md,
)
from financeharness.runtime.tool_registry import ToolSessionState
from financeharness.tools.core import build_load_skill_spec
from financeharness.tools.research import (
  FetchCache,
  build_equity_research_registry,
  default_skill_registry,
)
from tests.support import BUNDLED_SKILLS, chat_profile, run, runtime_config

DESCRIPTION = "Compare two tickers on valuation and recent price trend."


def skill_md(name="pair-check", description=DESCRIPTION, body="Step one.", extra=""):
  return f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n"


def write_skill(root, name, **kwargs):
  directory = root / name
  directory.mkdir(parents=True)
  (directory / "SKILL.md").write_text(skill_md(name=name, **kwargs), encoding="utf-8")


def test_skill_file_is_parsed_into_metadata_and_body():
  text = skill_md(extra="tags: [equity, pairs]\nrequires_tools:\n  - data_equity_prices\n")
  spec = parse_skill_md(text)
  assert spec.name == "pair-check"
  assert spec.description == DESCRIPTION
  assert spec.tags == ("equity", "pairs")
  assert spec.requires_tools == ("data_equity_prices",)
  assert spec.body == "Step one."
  assert spec.catalog_line() == f"- `pair-check`: {DESCRIPTION} Tags: equity, pairs."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Step one.", "must start with"),
        ("---\nname: pair-check\n", "not closed"),
        ("---\n- a\n- b\n---\nbody", "must be a mapping"),
        (skill_md(extra="version: 2\n"), "unknown keys"),
        (skill_md(name="Pair_Check"), "kebab-case"),
        (skill_md(description="Too short."), "description"),
        (skill_md(body=""), "body is empty"),
        (skill_md(extra="tags: 3\n"), "must be a string or list"),
        ("---\ndescription: " + DESCRIPTION + "\n---\nbody", "string `name`"),
    ],
)
def test_malformed_skill_file_is_rejected_with_the_reason(text, message):
  with pytest.raises(FrontmatterError, match=message):
    parse_skill_md(text)


def test_skill_name_must_match_its_directory(tmp_path):
  (tmp_path / "other-name").mkdir()
  (tmp_path / "other-name" / "SKILL.md").write_text(skill_md(), encoding="utf-8")
  with pytest.raises(FrontmatterError, match="!= directory"):
    load_skill_registry(tmp_path)


def test_later_root_overrides_a_skill_by_name_and_skips_broken_ones(tmp_path):
  base, user = tmp_path / "base", tmp_path / "user"
  write_skill(base, "pair-check", body="Base recipe.")
  write_skill(base, "kept-skill", body="Untouched.")
  write_skill(user, "pair-check", body="User recipe.")
  write_skill(user, "broken-skill", description="short")
  (user / "notes").mkdir()

  reg = load_skill_registry(base)
  merged = merge_skills(reg, user)

  assert merged == 1
  assert reg.names() == ["kept-skill", "pair-check"]
  assert reg.get("pair-check").body == "User recipe."


def test_broken_skill_fails_a_strict_load(tmp_path):
  write_skill(tmp_path, "broken-skill", description="short")
  with pytest.raises(FrontmatterError):
    load_skill_registry(tmp_path)


def test_missing_root_adds_nothing(tmp_path):
  assert merge_skills(SkillRegistry(), tmp_path / "absent") == 0


def test_default_registry_adds_project_and_env_skills_over_the_bundled_ones(tmp_path, monkeypatch):
  project, extra = tmp_path / "project", tmp_path / "extra"
  write_skill(project / "skills", "pair-check", body="Project recipe.")
  write_skill(project / "skills", "project-only", body="Project only recipe.")
  write_skill(extra, "pair-check", body="Env recipe.")
  write_skill(extra, "dcf-valuation", body="Replaced recipe.")
  monkeypatch.chdir(project)
  monkeypatch.setenv("FH_SKILLS_DIR", str(extra))

  reg = default_skill_registry()

  assert reg.get("project-only").body == "Project only recipe."
  assert reg.get("pair-check").body == "Env recipe."
  assert reg.get("dcf-valuation").body == "Replaced recipe."
  assert set(load_skill_registry(BUNDLED_SKILLS).names()) <= set(reg.names())


def test_bundled_skills_require_only_tools_that_exist(tmp_path):
  tools = build_equity_research_registry(
      FetchCache(), chat_profile(), config=runtime_config(tmp_path)
  )
  skills = load_skill_registry(BUNDLED_SKILLS)
  assert skills.names()
  for skill in skills.all():
    missing = [t for t in skill.requires_tools if tools.get(t) is None]
    assert not missing, f"{skill.name} requires unknown tools {missing}"


@pytest.fixture
def loader(tmp_path):
  tools = build_equity_research_registry(
      FetchCache(), chat_profile(), config=runtime_config(tmp_path)
  )
  skills = SkillRegistry()
  skills.register(
      parse_skill_md(
          skill_md(extra="requires_tools: [compute_valuation_dcf, not_a_tool]\n")
      )
  )
  skill_state, tool_state = SkillSessionState(), ToolSessionState()
  tools.register(build_load_skill_spec(skill_state, tool_state, skills, tools))
  return tools, skill_state, tool_state


def test_load_skill_returns_the_recipe_and_makes_its_tools_callable(loader):
  tools, skill_state, tool_state = loader

  result = run(dispatch("load_skill", {"skills": ["pair-check", "nope"]}, registry=tools))

  assert result.ok
  assert result.structured == {"loaded": ["pair-check"], "unknown": ["nope"]}
  assert "Step one." in result.markdown
  assert "Available: ['pair-check']" in result.markdown
  assert skill_state.loaded == {"pair-check"}
  assert tool_state.loaded_deferred == {"compute_valuation_dcf"}
  offered = [s["function"]["name"] for s in tool_state.visible_schemas(tools)]
  assert "compute_valuation_dcf" in offered
