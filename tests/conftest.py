"""Suite-wide fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path_factory, monkeypatch):
    """Point the user config directory at an empty one, away from the real home."""
    monkeypatch.setenv("FH_CONFIG_DIR", str(tmp_path_factory.mktemp("config")))
