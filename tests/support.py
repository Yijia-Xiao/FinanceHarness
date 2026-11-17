"""Test doubles and builders shared by the suite. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
  ChatCompletionMessageToolCall,
  Function,
)

import financeharness
from financeharness.providers.profiles import ModelProfile
from financeharness.runtime.config import load_runtime_config
from financeharness.runtime.tool_registry import ToolResponse, ToolSpec

BUNDLED_SKILLS = Path(financeharness.__file__).parent / "skills"


def run(coro):
  """Run a coroutine to completion."""
  return asyncio.run(coro)


def make_tool(name, schema, handler, *, tier="core"):
  """Build a tool spec around an async handler."""
  return ToolSpec(
      name=name,
      display_name=name.replace("_", "."),
      tier=tier,
      description=f"Test tool {name}, registered only inside the test suite.",
      request_schema=schema,
      handler=handler,
  )


def echo_handler(req):
  """Handler body that returns the validated request as the structured payload."""
  return ToolResponse(markdown="ok", structured=req.model_dump())


def runtime_config(tmp_path, **overrides):
  """Runtime config built from defaults only: no config file, no environment."""
  return load_runtime_config(tmp_path / "absent.json", env={}, overrides=overrides)


def chat_profile():
  """Profile that routes through the Chat Completions provider."""
  return ModelProfile(name="test", model="test-model")


def _completion(message, finish_reason):
  return ChatCompletion(
      id="test",
      choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
      created=0,
      model="test-model",
      object="chat.completion",
  )


def text_reply(content, finish_reason="stop"):
  """Model reply that carries text and no tool calls."""
  return _completion(ChatCompletionMessage(role="assistant", content=content), finish_reason)


def tool_reply(*calls):
  """Model reply that requests tools. Each call is ``(call_id, name, args_dict)``."""
  tool_calls = [
      ChatCompletionMessageToolCall(
          id=call_id,
          type="function",
          function=Function(name=name, arguments=json.dumps(args)),
      )
      for call_id, name, args in calls
  ]
  message = ChatCompletionMessage(role="assistant", content=None, tool_calls=tool_calls)
  return _completion(message, "tool_calls")


class ScriptedChatClient:
  """Stands in for the OpenAI client: replays scripted replies and records each request.

  Scripted exceptions are raised instead of returned.
  """

  def __init__(self, replies):
    self._replies = list(replies)
    self.requests = []
    self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

  async def _create(self, **kwargs):
    self.requests.append(kwargs)
    reply = self._replies.pop(0)
    if isinstance(reply, Exception):
      raise reply
    return reply

  def tool_names(self, index):
    """Names of the tools offered to the model in request ``index``."""
    return {t["function"]["name"] for t in self.requests[index].get("tools", [])}


class ScriptedGeminiClient:
  """Stands in for the google-genai client. Function calls carry no id, text ends the run."""

  def __init__(self, replies):
    self._replies = list(replies)
    self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

  async def _generate(self, **_kwargs):
    reply = self._replies.pop(0)
    if isinstance(reply, str):
      part = SimpleNamespace(function_call=None, text=reply, thought=False)
    else:
      name, args = reply
      part = SimpleNamespace(
          function_call=SimpleNamespace(id=None, name=name, args=args),
          thought_signature=None,
      )
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")
    return SimpleNamespace(candidates=[candidate])


class ApiError(Exception):
  """Provider error that carries an HTTP status, as SDK errors do."""

  def __init__(self, status_code, message):
    super().__init__(message)
    self.status_code = status_code
