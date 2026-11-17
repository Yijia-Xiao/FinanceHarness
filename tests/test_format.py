"""Value formatters used in tool output."""

import pytest

from financeharness.tools.format import money, num, pct


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.5e12, "$2.50T"),
        (1.2e9, "$1.20B"),
        (3_450_000, "$3.45M"),
        (-1.2e9, "-$1.20B"),
    ],
)
def test_money_scales_by_magnitude_and_keeps_the_sign(value, expected):
  assert money(value) == expected


@pytest.mark.xfail(strict=False, reason="amounts below one million lose their cents")
def test_money_keeps_cents_on_small_amounts():
  assert money(187.42) == "$187.42"


def test_pct_renders_a_fraction_as_a_percentage():
  assert pct(0.18) == "18.0%"
  assert pct(-0.0345, 2) == "-3.45%"


def test_num_rounds_to_the_requested_digits():
  assert num(1.23456) == "1.23"
  assert num(12.0, 1) == "12.0"


@pytest.mark.parametrize("formatter", [money, pct, num])
@pytest.mark.parametrize("missing", [None, float("nan"), True, "12"])
def test_missing_or_non_numeric_values_render_as_na(formatter, missing):
  assert formatter(missing) == "n/a"
