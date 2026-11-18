"""Failed model calls end the run with their cause."""

import pytest

from financeharness import cli
from financeharness.providers import gemini as gemini_module
from financeharness.providers.profiles import ModelProfile, load_profiles
from financeharness.runtime.agent import Agent
from financeharness.runtime.recovery import describe_model_error, missing_api_key
from financeharness.runtime.tool_registry import ToolRegistry
from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
from tests.support import ApiError, run, runtime_config, text_reply
from tests.test_agent import calc_agent


def failed_run(tmp_path, replies, **config):
    agent, client = calc_agent(tmp_path, replies, **config)
    errors = []

    def on_event(kind, data):
        if kind == "error":
            errors.append(data)

    trajectory = run(agent.run("Question", on_event=on_event))
    assert trajectory["termination"] == "error"
    assert errors == [{"error": trajectory["error"]}]
    return trajectory["error"], client


def test_rejected_credentials_name_the_key_to_check(tmp_path):
    error, _ = failed_run(tmp_path, [ApiError(401, "Incorrect API key provided")])
    assert error == (
        "the API rejected the credentials (HTTP 401, profile 'test', model 'test-model'):"
        " check OPENAI_API_KEY. Incorrect API key provided"
    )


def test_unknown_model_names_the_profile_and_model(tmp_path):
    error, _ = failed_run(tmp_path, [ApiError(404, "The model `test-model` does not exist")])
    assert error.startswith("the model or endpoint was not found (HTTP 404, profile 'test', model")
    assert "check the model name and base URL" in error
    assert "does not exist" in error


def test_other_failures_report_the_error_type_and_message(tmp_path):
    error, _ = failed_run(tmp_path, [ValueError("unsupported\n  parameter: top_k")])
    assert error == (
        "the model call failed (ValueError, profile 'test', model 'test-model'):"
        " unsupported parameter: top_k"
    )


def test_exhausted_retries_report_the_last_error_and_the_retry_count(tmp_path):
    error, client = failed_run(
        tmp_path, [ApiError(503, "overloaded")] * 3, limits={"max_transient_retries": 2}
    )
    assert len(client.requests) == 3
    assert error == (
        "the model call failed (ApiError, HTTP 503, profile 'test', model 'test-model'):"
        " overloaded (gave up after 2 retries)"
    )


def test_missing_api_key_is_reported_before_any_client_is_built(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    built = []
    monkeypatch.setattr(gemini_module, "_build_client", lambda profile: built.append(profile))
    registry = ToolRegistry()
    registry.register(CALC_SPEC)
    agent = Agent(
        profile=load_profiles(tmp_path / "absent.json", env={})["gemini"],
        registry=registry,
        config=runtime_config(tmp_path),
    )

    trajectory = run(agent.run("Question"))

    assert trajectory["termination"] == "error"
    assert trajectory["error"] == "no API key for profile 'gemini': set GEMINI_API_KEY"
    assert built == []


def test_successful_run_has_no_error(tmp_path):
    agent, _ = calc_agent(tmp_path, [text_reply("Final report.")])
    assert run(agent.run("Question"))["error"] is None


def test_keyless_endpoint_is_not_told_to_check_an_environment_variable():
    profile = ModelProfile(name="local", model="m", api_key_literal="EMPTY")
    message = describe_model_error(ApiError(403, "forbidden"), profile)
    assert "check the endpoint's access settings" in message
    assert "API_KEY" not in message


def test_endpoints_that_may_run_without_a_key_are_not_blocked():
    hosted = ModelProfile(name="gpt", model="m")
    local = ModelProfile(name="local", model="m", base_url="http://127.0.0.1:9/v1")
    literal = ModelProfile(name="lit", model="m", api_key_literal="EMPTY")
    gemini = ModelProfile(name="gemini", model="m", provider="gemini", api_key_env="GEMINI_API_KEY")

    assert missing_api_key(hosted, env={}) == "OPENAI_API_KEY"
    assert missing_api_key(hosted, env={"OPENAI_API_KEY": "k"}) is None
    assert missing_api_key(local, env={}) is None
    assert missing_api_key(literal, env={}) is None
    assert missing_api_key(gemini, env={}) == "GEMINI_API_KEY"
    assert missing_api_key(gemini, env={"GOOGLE_API_KEY": "k"}) is None
    assert missing_api_key(gemini, env={"GEMINI_API_KEY": "k"}) is None


def test_empty_response_from_the_model_is_reported_as_the_cause(tmp_path):
    no_choices = text_reply("x")
    no_choices.choices = []
    error, _ = failed_run(tmp_path, [no_choices])
    assert error == "the model returned no output (an empty or blocked response)"


def test_error_without_a_message_still_says_something(tmp_path):
    error, client = failed_run(
        tmp_path, [ConnectionError()] * 2, limits={"max_transient_retries": 1}
    )
    assert len(client.requests) == 2
    assert error == (
        "the model call failed (ConnectionError, profile 'test', model 'test-model'):"
        " no details given (gave up after 1 retry)"
    )


def test_key_like_text_in_an_api_message_is_masked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "plainsecret-123456")
    message = "invalid token 'sk-live-SECRETSECRET' and plainsecret-123456 and AIzaSyA1234567890"
    error, _ = failed_run(tmp_path, [ApiError(401, message)])
    assert "SECRETSECRET" not in error
    assert "plainsecret-123456" not in error
    assert "A1234567890" not in error
    assert "invalid token 'sk-<masked>' and <api key> and AIza<masked>" in error


def test_invalid_key_reported_with_http_400_is_not_retried(tmp_path):
    reply = ApiError(400, "API key not valid. Please pass a valid API key.")
    error, client = failed_run(tmp_path, [reply])
    assert len(client.requests) == 1
    assert error.startswith("the API rejected the credentials (HTTP 400, profile 'test'")
    assert "check OPENAI_API_KEY" in error


def test_quiet_cli_run_prints_the_cause(monkeypatch, capsys):
    async def failed_research(question, **kwargs):
        return {
            "termination": "error",
            "error": "no API key for profile 'gemini': set GEMINI_API_KEY",
            "rounds": 1,
            "citations": [],
            "elapsed_s": 0.0,
            "prediction": "",
        }

    monkeypatch.setattr(cli, "run_research", failed_research)
    monkeypatch.setattr("sys.argv", ["fh", "-p", "question", "--quiet"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "[fh] error: no API key for profile 'gemini': set GEMINI_API_KEY" in err
