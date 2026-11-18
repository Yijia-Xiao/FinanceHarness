"""Events and trajectories hold strict JSON only."""

import json

import pytest

from financeharness.runtime.agent import Agent
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
from financeharness.tools.data.equity.common import is_num
from tests.support import (
    ScriptedChatClient,
    chat_profile,
    run,
    runtime_config,
    text_reply,
    tool_reply,
)


def test_events_and_trajectory_stay_strict_json_when_the_model_writes_nan(tmp_path):
    # a tool call whose argument text holds a bare NaN, as some models emit
    nan_call = tool_reply(("c1", "calc", {"expression": "1+1"}))
    nan_call.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"expression": "1+1", "precision": NaN}'
    client = ScriptedChatClient(
        [
            nan_call,
            tool_reply(("c2", "calc", {"expression": "(-8) ** (1 / 3)"})),
            text_reply("Done."),
        ]
    )
    registry = ToolRegistry()
    registry.register(CALC_SPEC)
    agent = Agent(
        profile=chat_profile(), registry=registry, config=runtime_config(tmp_path), client=client
    )
    events = []

    trajectory = run(agent.run("Question", on_event=lambda kind, data: events.append(data)))

    json.dumps(events, allow_nan=False)
    json.dumps(trajectory, allow_nan=False)
    assert [entry["structured"] for entry in trajectory["tool_log"]] == [
        {"error": "args_not_json"},
        {"error": "tool_error"},
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(float("nan"), False), (float("inf"), False), (1.5, True), (True, False), (None, False)],
)
def test_data_tool_number_guard_accepts_only_finite_numbers(value, expected):
    assert is_num(value) is expected
