"""Tool-call ids: unique within a run whatever the API sends."""

from financeharness.providers.client import _assemble_completion
from tests.support import run, text_reply, tool_reply
from tests.test_agent import calc_agent, tool_messages


def test_chat_tool_calls_without_ids_get_distinct_ids_within_and_across_rounds(tmp_path):
    agent, client = calc_agent(
        tmp_path,
        [
            tool_reply(("", "calc", {"expression": "1+1"}), ("", "calc", {"expression": "2+2"})),
            tool_reply(("", "calc", {"expression": "3+3"})),
            text_reply("Done."),
        ],
    )
    trajectory = run(agent.run("Add."))
    ids = [m["tool_call_id"] for m in tool_messages(trajectory)]
    assert ids == ["call_1", "call_2", "call_3"]
    requested = [tc["id"] for m in trajectory["messages"] for tc in m.get("tool_calls") or []]
    assert requested == ids
    assert "prev:call_3." in tool_messages(trajectory)[2]["content"]


TWO = {"expression": "2+2"}
ONE = {"expression": "1"}


def test_id_repeated_by_the_api_in_a_later_round_is_replaced(tmp_path):
    agent, _ = calc_agent(
        tmp_path,
        [
            tool_reply(("call_0", "calc", {"expression": "1+1"})),
            tool_reply(("call_0", "calc", TWO), ("x9", "calc", ONE)),
            text_reply("Done."),
        ],
    )
    trajectory = run(agent.run("Add."))
    ids = [m["tool_call_id"] for m in tool_messages(trajectory)]
    assert ids == ["call_0", "call_3", "x9"]


def test_earlier_result_stays_reachable_when_the_api_sends_no_ids(tmp_path):
    agent, _ = calc_agent(
        tmp_path,
        [
            tool_reply(("", "calc", {"expression": "40+2"})),
            tool_reply(("", "calc", {"expression": "1+1"})),
            tool_reply(("", "calc", {"expression": "prev:call_1.expression"})),
            text_reply("Done."),
        ],
    )
    trajectory = run(agent.run("Add."))
    assert "**42**" in tool_messages(trajectory)[2]["content"]


def test_ids_already_in_the_conversation_history_are_not_reused(tmp_path):
    agent, _ = calc_agent(
        tmp_path, [tool_reply(("", "calc", {"expression": "1+1"})), text_reply("Done.")]
    )
    history = [
        {"role": "user", "content": "Earlier."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "x"},
        {"role": "assistant", "content": "Earlier answer."},
    ]
    trajectory = run(agent.run("Add.", history=history))
    assert tool_messages(trajectory)[-1]["tool_call_id"] == "call_2"


def test_streamed_tool_call_without_an_id_is_left_for_the_loop_to_name():
    streamed = {0: {"id": "", "name": "calc", "arguments": "{}", "extra_content": None}}
    completion = _assemble_completion(
        resp_id="r",
        model="m",
        created=0,
        content="",
        tool_calls=streamed,
        finish_reason="tool_calls",
    )
    assert completion.choices[0].message.tool_calls[0].id == ""


def test_id_that_cannot_be_written_in_a_reference_is_replaced(tmp_path):
    agent, _ = calc_agent(
        tmp_path,
        [
            tool_reply(("functions.calc:0", "calc", {"expression": "40+2"}), (" ", "calc", ONE)),
            tool_reply(("ok_id-7", "calc", {"expression": "prev:call_1.expression"})),
            text_reply("Done."),
        ],
    )
    trajectory = run(agent.run("Add."))
    results = tool_messages(trajectory)
    assert [m["tool_call_id"] for m in results] == ["call_1", "call_2", "ok_id-7"]
    assert "**42**" in results[2]["content"]


def test_id_sent_by_the_api_is_kept_when_an_earlier_call_in_the_turn_has_none(tmp_path):
    agent, _ = calc_agent(
        tmp_path,
        [tool_reply(("", "calc", ONE), ("call_1", "calc", TWO)), text_reply("Done.")],
    )
    trajectory = run(agent.run("Add."))
    assert [m["tool_call_id"] for m in tool_messages(trajectory)] == ["call_2", "call_1"]


def test_ids_of_tool_results_in_the_history_are_not_reused(tmp_path):
    agent, _ = calc_agent(tmp_path, [tool_reply(("", "calc", ONE)), text_reply("Done.")])
    history = [
        {"role": "user", "content": "Earlier."},
        {"role": "tool", "tool_call_id": "call_1", "content": "x"},
        {"role": "assistant", "content": "Earlier answer."},
    ]
    trajectory = run(agent.run("Add.", history=history))
    assert tool_messages(trajectory)[-1]["tool_call_id"] == "call_2"
