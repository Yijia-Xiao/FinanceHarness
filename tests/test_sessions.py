"""Session store: one JSON file per session, committed only for answered turns."""

import itertools
import json

import pytest

from financeharness.service import sessions
from financeharness.service.sessions import SessionStore, default_sessions_dir


def turn(question, answer="Answer.", termination="answer"):
    return {
        "termination": termination,
        "messages": [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


@pytest.fixture
def clock(monkeypatch):
    ticks = (f"2026-01-01T00:00:{n:02d}+00:00" for n in itertools.count())
    monkeypatch.setattr(sessions, "_now", lambda: next(ticks))


@pytest.fixture
def store(tmp_path, clock):
    return SessionStore(tmp_path / "sessions")


def test_answered_turn_is_stored_without_the_system_prompt(store):
    store.commit("s1", turn("What moved ACME today?"))
    assert "s1" in store
    assert store.history("s1") == [
        {"role": "user", "content": "What moved ACME today?"},
        {"role": "assistant", "content": "Answer."},
    ]


@pytest.mark.parametrize("termination", ["error", "max_rounds", "timeout", "length", ""])
def test_turn_that_did_not_answer_is_not_stored(store, termination):
    store.commit("s1", turn("What moved ACME today?"))
    store.commit("s1", turn("Follow up", termination=termination))
    assert store.history("s1") == [
        {"role": "user", "content": "What moved ACME today?"},
        {"role": "assistant", "content": "Answer."},
    ]


def test_later_turns_keep_the_creation_time_and_count_user_turns(store):
    first = turn("What moved ACME today?")
    second = {
        "termination": "answer",
        "messages": [*first["messages"], {"role": "user", "content": "And yesterday?"}],
    }
    store.commit("s1", first)
    store.commit("s1", second)

    (summary,) = store.list()
    assert summary == {
        "id": "s1",
        "title": "What moved ACME today?",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:02+00:00",
        "turns": 2,
    }


def test_long_first_question_is_shortened_for_the_title(store):
    store.commit("s1", turn("x" * 200))
    (summary,) = store.list()
    assert summary["title"].startswith("x" * 80)
    assert len(summary["title"]) == 81


def test_sessions_are_listed_most_recently_updated_first(store):
    store.commit("old", turn("First"))
    store.commit("new", turn("Second"))
    store.commit("old", turn("First again"))
    assert [s["id"] for s in store.list()] == ["old", "new"]


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "a.b", "", ".hidden", "x" * 129, None])
def test_unsafe_session_id_never_reaches_the_filesystem(tmp_path, store, session_id):
    store.commit(session_id, turn("Question"))
    store.replace(session_id, [{"role": "user", "content": "Question"}])
    assert session_id not in store
    assert store.history(session_id) is None
    assert list(tmp_path.rglob("*.json")) == []


def test_corrupt_or_missing_session_reads_as_absent(tmp_path, store):
    store.commit("good", turn("Question"))
    (tmp_path / "sessions" / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "sessions" / "list.json").write_text("[1]", encoding="utf-8")
    assert store.history("bad") is None
    assert store.history("list") is None
    assert store.history("missing") is None
    assert [s["id"] for s in store.list()] == ["good"]


def test_replace_swaps_the_stored_history(store):
    store.commit("s1", turn("What moved ACME today?"))
    store.replace("s1", [{"role": "user", "content": "Summary of the session so far."}])
    assert store.history("s1") == [{"role": "user", "content": "Summary of the session so far."}]
    assert store.list()[0]["title"] == "What moved ACME today?"


def test_session_file_holds_non_ascii_text_as_utf8_and_leaves_no_temp_file(tmp_path, store):
    store.commit("s1", turn("What is Nestlé's margin?", answer="About 17%, per Nestlé."))
    directory = tmp_path / "sessions"
    assert [p.name for p in directory.iterdir()] == ["s1.json"]
    record = json.loads((directory / "s1.json").read_text(encoding="utf-8"))
    assert record["messages"][1]["content"] == "About 17%, per Nestlé."
    assert "Nestlé" in (directory / "s1.json").read_text(encoding="utf-8")


def test_store_listing_is_empty_before_the_directory_exists(store):
    assert store.list() == []


def test_new_ids_are_safe_and_distinct(store):
    first, second = store.new_id(), store.new_id()
    assert first != second
    store.commit(first, turn("Question"))
    assert first in store


def test_sessions_directory_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FH_SESSIONS_DIR", str(tmp_path / "custom"))
    assert default_sessions_dir() == tmp_path / "custom"
    SessionStore().commit("s1", turn("Question"))
    assert (tmp_path / "custom" / "s1.json").is_file()
