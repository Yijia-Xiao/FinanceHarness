"""Location and reading of the user's configuration files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

CONFIG_DIR_ENV = "FH_CONFIG_DIR"
# Messages already written by ``warn_once``.
_warned: set[str] = set()


def warn_once(message):
    """Write a configuration warning to stderr, once per process."""
    if message in _warned:
        return
    _warned.add(message)
    print(f"[fh] {message}", file=sys.stderr)


def reason_of(err):
    """One line that says why a config file or entry cannot be used."""
    if isinstance(err, ValidationError):
        first = err.errors()[0]
        return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return str(err) or type(err).__name__


def config_dir(env=None):
    """The user config directory: ``$FH_CONFIG_DIR``, else ``~/.financeharness``.

    ``env`` supplies ``FH_CONFIG_DIR`` only; the home directory is always the
    process's own. Returns None when no home directory can be determined.
    """
    env = os.environ if env is None else env
    override = (env.get(CONFIG_DIR_ENV) or "").strip()
    try:
        if override:
            return Path(override).expanduser()
        return Path.home() / ".financeharness"
    except RuntimeError:
        return None


def config_file(name, env=None):
    """Path of the user config file ``name`` (it need not exist), or None."""
    directory = config_dir(env)
    return directory / name if directory is not None else None


def read_json_object(path):
    """The JSON object in the config file at ``path``; None when there is no file.

    Raises OSError or ValueError when the file exists but cannot be read, is not
    JSON, or is not an object at the top level.
    """
    if path is None:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("the top level is not a JSON object")
    return data
