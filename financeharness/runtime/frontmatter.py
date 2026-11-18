"""Minimal YAML-frontmatter parsing for skill specs.

A SKILL.md is ``---\\n<yaml>\\n---\\n<body>``. This splits and validates the
frontmatter; shared so other spec families reuse it.
"""

from __future__ import annotations

import yaml


class FrontmatterError(ValueError):
    """A malformed spec file."""


def split_frontmatter(text, *, source=None):
    """Return ``(frontmatter_dict, body)``.

    Raises FrontmatterError if the file lacks a leading ``---`` fenced YAML block.
    """
    where = f" ({source})" if source else ""
    if not text.lstrip().startswith("---"):
        raise FrontmatterError(f"spec{where} must start with a '---' frontmatter block")
    stripped = text.lstrip()
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        raise FrontmatterError(f"spec{where} frontmatter block is not closed with '---'")
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"spec{where} has invalid YAML frontmatter: {exc}") from exc
    if not isinstance(fm, dict):
        raise FrontmatterError(f"spec{where} frontmatter must be a mapping")
    return fm, parts[2].strip()


def as_str_list(fm, key):
    """Coerce a frontmatter value to a tuple of strings ([] when absent)."""
    v = fm.get(key)
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    if isinstance(v, list):
        return tuple(str(x) for x in v)
    raise FrontmatterError(f"frontmatter `{key}` must be a string or list")
