"""Agent loop against a scripted model client: tool rounds, loading, limits and recovery."""

import pytest

from financeharness.providers.profiles import load_profiles
from financeharness.runtime.agent import Agent
from financeharness.runtime.skill_registry import load_skill_registry
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
from financeharness.tools.research import (
    FetchCache,
    build_equity_research_registry,
    citation_finalizer,
)
from tests.support import (
    BUNDLED_SKILLS,
    ApiError,
    ScriptedChatClient,
    ScriptedGeminiClient,
    chat_profile,
    run,
    runtime_config,
    text_reply,
    tool_reply,
)

CORE_TOOLS = {
    "search",
    "visit",
    "compose_citations",
    "calc",
    "update_plan",
    "load_tool",
    "load_skill",
}


def calc_agent(tmp_path, replies, **config):
    """Agent with the calc tool only, driven by scripted replies."""
    config.setdefault("backoff", {"base_delay_s": 0, "jitter": False})
    client = ScriptedChatClient(replies)
    registry = ToolRegistry()
    registry.register(CALC_SPEC)
    agent = Agent(
        profile=chat_profile(),
        registry=registry,
        config=runtime_config(tmp_path, **config),
        client=client,
    )
    return agent, client


def tool_messages(trajectory):
    return [m for m in trajectory["messages"] if m["role"] == "tool"]


def test_run_loads_deferred_tools_chains_results_and_cites_sources(tmp_path):
    client = ScriptedChatClient(
        [
            tool_reply(
                ("a1", "load_tool", {"tools": ["compute_valuation_wacc", "compute_valuation_dcf"]})
            ),
            tool_reply(
                (
                    "a2",
                    "compute_valuation_wacc",
                    {"risk_free_rate": 0.04, "equity_risk_premium": 0.05, "beta": 1.2},
                )
            ),
            tool_reply(
                (
                    "a3",
                    "compute.valuation.dcf",
                    {
                        "fcf_schedule": "[100, 110]",
                        "discount_rate": "prev:a2.wacc",
                        "terminal_growth_rate": 0.02,
                    },
                )
            ),
            text_reply("Enterprise value is about 1,341 [1]. Peers trade lower [4]."),
        ]
    )
    cache = FetchCache()
    cache.add_citation("https://example.com/10k", "Annual report")
    config = runtime_config(tmp_path)
    agent = Agent(
        profile=chat_profile(),
        registry=build_equity_research_registry(
            cache, chat_profile(), client=client, config=config
        ),
        config=config,
        client=client,
        finalize=citation_finalizer(cache),
        skill_registry=load_skill_registry(BUNDLED_SKILLS),
    )
    events = []

    trajectory = run(agent.run("Value ACME.", on_event=lambda kind, data: events.append(kind)))

    assert client.tool_names(0) == CORE_TOOLS
    assert client.tool_names(1) == CORE_TOOLS | {"compute_valuation_wacc", "compute_valuation_dcf"}
    assert [entry["ok"] for entry in trajectory["tool_log"]] == [True, True, True]
    dcf = trajectory["tool_log"][2]["structured"]
    assert dcf["discount_rate"] == pytest.approx(0.10)
    assert dcf["enterprise_value"] == pytest.approx(1340.9090909)
    assert trajectory["termination"] == "answer"
    assert trajectory["rounds"] == 4
    assert trajectory["prediction"] == (
        "Enterprise value is about 1,341 [1]. Peers trade lower.\n\n"
        "## References\n\n[1] Annual report\n    https://example.com/10k\n"
    )
    assert trajectory["citation_stats"]["n_orphans_stripped"] == 1
    assert events.count("tool_call") == events.count("tool_result") == 3
    assert events[-2:] == ["answer", "done"]


def test_tool_result_is_sent_back_to_the_model_under_its_call_id(tmp_path):
    agent, client = calc_agent(
        tmp_path, [tool_reply(("c1", "calc", {"expression": "6*7"})), text_reply("It is 42.")]
    )
    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]

    trajectory = run(agent.run("What is 6 times 7?", history=history))

    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert sent[3]["content"] == "What is 6 times 7?"
    assert sent[4]["tool_calls"][0]["id"] == "c1"
    assert sent[5]["tool_call_id"] == "c1"
    assert sent[5]["content"].startswith("`6*7` = **42**")
    assert trajectory["prediction"] == "It is 42."


def test_failed_tool_call_is_reported_to_the_model_and_the_run_continues(tmp_path):
    agent, client = calc_agent(
        tmp_path,
        [
            tool_reply(("c1", "price_lookup", {"ticker": "ACME"}), ("c2", "calc", {})),
            text_reply("I could not look that up."),
        ],
    )

    trajectory = run(agent.run("Price of ACME?"))

    first, second = tool_messages(trajectory)
    assert "Unknown tool: 'price_lookup'" in first["content"]
    assert "expression: Field required" in second["content"]
    assert [entry["ok"] for entry in trajectory["tool_log"]] == [False, False]
    assert trajectory["termination"] == "answer"


def test_run_stops_at_the_round_limit(tmp_path):
    replies = [tool_reply((f"c{n}", "calc", {"expression": "1+1"})) for n in range(5)]
    agent, client = calc_agent(tmp_path, replies, limits={"max_rounds": 2})

    trajectory = run(agent.run("Loop forever."))

    assert trajectory["termination"] == "max_rounds"
    assert trajectory["rounds"] == 2
    assert len(client.requests) == 2
    assert trajectory["prediction"] == ""


def test_transient_model_error_is_retried(tmp_path):
    agent, client = calc_agent(
        tmp_path, [ApiError(503, "overloaded"), ApiError(429, "slow down"), text_reply("Done.")]
    )
    trajectory = run(agent.run("Question"))
    assert trajectory["termination"] == "answer"
    assert trajectory["rounds"] == 1
    assert len(client.requests) == 3


def test_transient_model_errors_end_the_run_after_the_retry_limit(tmp_path):
    agent, client = calc_agent(
        tmp_path,
        [ApiError(503, "overloaded")] * 3,
        limits={"max_transient_retries": 2},
    )
    trajectory = run(agent.run("Question"))
    assert trajectory["termination"] == "error"
    assert len(client.requests) == 3


def test_rejected_credentials_end_the_run_without_a_retry(tmp_path):
    agent, client = calc_agent(tmp_path, [ApiError(401, "Incorrect API key provided")])
    trajectory = run(agent.run("Question"))
    assert trajectory["termination"] == "error"
    assert len(client.requests) == 1


def test_failed_model_call_reports_its_cause(tmp_path):
    agent, _ = calc_agent(tmp_path, [ApiError(401, "Incorrect API key provided")])
    errors = []

    def on_event(kind, data):
        if kind == "error":
            errors.append(data["error"])

    run(agent.run("Question", on_event=on_event))

    assert any("Incorrect API key provided" in error for error in errors)


def test_truncated_reply_is_requested_again_with_a_larger_budget(tmp_path):
    agent, client = calc_agent(
        tmp_path,
        [text_reply("Partial", finish_reason="length"), text_reply("Complete report.")],
        generation={"max_tokens": 100, "max_tokens_ceiling": 400},
    )
    trajectory = run(agent.run("Question"))
    assert [r["max_tokens"] for r in client.requests] == [100, 200]
    assert trajectory["prediction"] == "Complete report."
    assert trajectory["termination"] == "answer"


def test_reply_still_truncated_at_the_ceiling_is_returned_as_truncated(tmp_path):
    agent, client = calc_agent(
        tmp_path,
        [text_reply("Partial", finish_reason="length")] * 2,
        generation={"max_tokens": 100, "max_tokens_ceiling": 200},
    )
    trajectory = run(agent.run("Question"))
    assert [r["max_tokens"] for r in client.requests] == [100, 200]
    assert trajectory["termination"] == "length"
    assert trajectory["prediction"] == "Partial"


def test_blank_reply_is_asked_again_and_the_next_answer_is_used(tmp_path):
    agent, client = calc_agent(tmp_path, [text_reply(""), text_reply("Final report.")])
    trajectory = run(agent.run("Question"))
    assert trajectory["termination"] == "answer"
    assert trajectory["prediction"].strip() == "Final report."
    assert len(client.requests) == 2
    assert client.requests[1]["messages"][-1]["role"] == "user"


def test_run_of_blank_replies_ends_as_empty_answer(tmp_path):
    agent, client = calc_agent(tmp_path, [text_reply(""), text_reply(" \n"), text_reply("")])
    trajectory = run(agent.run("Question"))
    assert trajectory["termination"] == "empty_answer"
    assert not trajectory["prediction"].strip()
    assert len(client.requests) == 3


def test_tool_calls_without_provider_ids_get_distinct_ids_across_rounds(tmp_path):
    client = ScriptedGeminiClient(
        [
            ("calc", {"expression": "1+1"}),
            ("calc", {"expression": "2+2"}),
            "Done.",
        ]
    )
    registry = ToolRegistry()
    registry.register(CALC_SPEC)
    agent = Agent(
        profile=load_profiles(tmp_path / "absent.json", env={})["gemini"],
        registry=registry,
        config=runtime_config(tmp_path),
        client=client,
    )

    trajectory = run(agent.run("Add twice."))

    ids = [m["tool_call_id"] for m in tool_messages(trajectory)]
    assert len(ids) == 2
    assert len(set(ids)) == 2
