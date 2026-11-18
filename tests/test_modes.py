"""Execution modes and their selection from the command line."""

import pytest

from financeharness import cli
from financeharness.runtime.modes import resolve_mode


@pytest.mark.parametrize(
    ("name", "expected"),
    [(None, "research"), ("", "research"), ("Analytical", "analytical"), ("auto", "auto")],
)
def test_mode_defaults_to_research_and_is_normalized(name, expected):
    assert resolve_mode(name) == expected


def test_cli_has_no_equity_flag(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["fh", "-p", "question", "--equity"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --equity" in capsys.readouterr().err
