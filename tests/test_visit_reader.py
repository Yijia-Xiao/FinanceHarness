"""Page reader: input sizing against the context window, retry on truncated output."""

import json

from financeharness.providers.profiles import ModelProfile, load_profiles
from financeharness.tools.research.visit_reader import page_char_limit, read_page
from tests.support import ScriptedChatClient, chat_profile, run, runtime_config, text_reply

SUMMARY = json.dumps({"accessible": True, "summary": "Revenue rose 8%.", "evidence": "p. 41"})


def read(tmp_path, replies, text="page text", profile=None, **overrides):
    client = ScriptedChatClient(replies)
    config = runtime_config(tmp_path, **overrides)
    result = run(
        read_page(profile or chat_profile(), text, goal="revenue", client=client, config=config)
    )
    return result, client


def page_sent(client, index=0):
    return client.requests[index]["messages"][1]["content"].split("Webpage content:\n\n")[1]


def test_page_is_cut_to_what_the_reader_context_window_holds(tmp_path):
    profile = ModelProfile(name="small", model="m", context_window=40_000)
    config = runtime_config(tmp_path)
    limit = page_char_limit(profile, config)
    assert limit == (40_000 - 2 * config.reader.max_tokens - 2048) * 3

    _, client = read(tmp_path, [text_reply(SUMMARY)], text="x" * (limit + 500), profile=profile)

    assert len(page_sent(client)) == limit


def test_profile_without_a_window_uses_the_runtime_context_window(tmp_path):
    config = runtime_config(tmp_path, recovery={"context_window": 50_000})
    assert page_char_limit(chat_profile(), config) == (50_000 - 16_384 - 2048) * 3


def test_whole_filing_fits_the_default_backbone(tmp_path):
    gemini = load_profiles(tmp_path / "absent.json", env={})["gemini"]
    assert page_char_limit(gemini, runtime_config(tmp_path)) > 800_000


def test_configured_character_limit_takes_precedence(tmp_path):
    _, client = read(tmp_path, [text_reply(SUMMARY)], text="x" * 900, reader={"max_chars": 100})
    assert len(page_sent(client)) == 100


def test_window_smaller_than_the_output_budget_still_sends_some_text(tmp_path):
    profile = ModelProfile(name="tiny", model="m", context_window=4096)
    assert page_char_limit(profile, runtime_config(tmp_path)) == 2048 * 3


def test_complete_reply_is_parsed_in_one_call(tmp_path):
    result, client = read(tmp_path, [text_reply(SUMMARY)])
    assert result == {"accessible": True, "summary": "Revenue rose 8%.", "evidence": "p. 41"}
    assert len(client.requests) == 1


def test_truncated_read_is_retried_once_with_a_larger_output_budget(tmp_path):
    result, client = read(tmp_path, [text_reply("", "length"), text_reply(SUMMARY)])
    assert result["summary"] == "Revenue rose 8%."
    assert [r["max_tokens"] for r in client.requests] == [8192, 16384]


def test_partial_summary_is_kept_when_the_retry_fails(tmp_path):
    partial = '{"accessible": true, "summary": "Revenue rose'
    result, client = read(
        tmp_path, [text_reply(partial, "length"), RuntimeError("context exceeded")]
    )
    assert result["accessible"]
    assert result["summary"].startswith("Revenue rose")
    assert "reader_error" not in result
    assert len(client.requests) == 2


def test_partial_summary_is_kept_when_the_retry_writes_nothing(tmp_path):
    partial = '{"accessible": true, "summary": "Revenue rose'
    result, client = read(tmp_path, [text_reply(partial, "length"), text_reply("", "length")])
    assert result["summary"].startswith("Revenue rose")
    assert len(client.requests) == 2


def test_read_that_never_writes_is_a_reader_error_after_one_retry(tmp_path):
    result, client = read(tmp_path, [text_reply("", "length"), text_reply("", "length")])
    assert not result["accessible"]
    assert "16384 output tokens" in result["reader_error"]
    assert len(client.requests) == 2


def test_failed_reader_call_is_a_reader_error(tmp_path):
    result, client = read(tmp_path, [RuntimeError("connection refused")])
    assert result == {
        "accessible": False,
        "summary": "",
        "evidence": "",
        "reader_error": "connection refused",
    }
    assert len(client.requests) == 1


def test_reply_without_a_choice_is_a_reader_error(tmp_path):
    empty = text_reply("x")
    empty.choices = []
    result, _ = read(tmp_path, [empty])
    assert not result["accessible"]
    assert result["reader_error"]


def test_no_retry_when_the_budget_is_already_at_the_ceiling(tmp_path):
    result, client = read(
        tmp_path,
        [text_reply("", "length")],
        reader={"max_tokens": 4096},
        generation={"max_tokens_ceiling": 4096},
    )
    assert "4096 output tokens" in result["reader_error"]
    assert len(client.requests) == 1
