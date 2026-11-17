"""Reader-model extraction helpers for the `visit` tool."""

from __future__ import annotations

import re

from financeharness.providers import complete
from financeharness.runtime.jsonutil import extract_json_obj
from financeharness.runtime.recovery import escalate_max_tokens

EXTRACTOR_PROMPT = """\
You extract grounded content from a fetched webpage for a research agent. Return \
ONLY a JSON object with three fields:
  "accessible": true if the text is real article/report content; false if it is a \
paywall, login/bot wall, cookie/consent page, error page, or essentially empty.
  "summary": a faithful, information-dense summary of the page's substantive \
content (figures, facts, claims), using the page's own numbers. Empty string if \
not accessible.
  "evidence": one or two short verbatim quotes that support the key facts. Empty \
string if not accessible.
Report only what the page actually states. If it carries no real content (paywall, \
bot-wall, error, or essentially empty), set accessible=false with empty summary/evidence."""


def _salvage_reader_fields(raw):
  """Pull reader fields from JSON-shaped text the parser could not load."""

  def field(name):
    # the closing quote is optional at the end of the text: a reply cut off
    # by the output limit ends inside a field
    m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|\Z)', raw, re.DOTALL)
    if not m:
      return ""
    return (
        m.group(1)
        .replace('\\"', '"')
        .replace("\\n", " ")
        .replace("\\t", " ")
        .strip()
    )

  summary, evidence = field("summary"), field("evidence")
  acc = re.search(r'"accessible"\s*:\s*(true|false)', raw)
  accessible = acc.group(1) == "true" if acc else bool(summary)
  return {"accessible": accessible, "summary": summary, "evidence": evidence}


# Sizing of the page text against the reader's context window: a conservative
# characters-per-token figure for filings (digits and tables tokenize densely)
# and room for the extraction prompt and the research goal.
_CHARS_PER_TOKEN = 3
_PROMPT_RESERVE_TOKENS = 2048


def _token_budgets(config):
  """Output budgets for a page read: the configured one, then one raised retry."""
  first = config.reader.max_tokens
  raised = escalate_max_tokens(
      first,
      config.generation.max_tokens_ceiling,
      config.generation.max_tokens_escalation_factor,
  )
  return [first] if raised is None else [first, raised]


def page_char_limit(profile, config):
  """Characters of page text the reader is sent.

  ``reader.max_chars`` when set; otherwise what fits the reader's context window
  (the profile's, else ``recovery.context_window``) beside the prompt and the
  largest output budget.
  """
  if config.reader.max_chars is not None:
    return config.reader.max_chars
  window = profile.context_window or config.recovery.context_window
  room = window - _token_budgets(config)[-1] - _PROMPT_RESERVE_TOKENS
  return max(room, _PROMPT_RESERVE_TOKENS) * _CHARS_PER_TOKEN


async def read_page(
    profile,
    text,
    *,
    goal,
    client,
    config,
):
  """Run the reader-model extraction over page text."""
  snippet = text[: page_char_limit(profile, config)]
  user = (
      f"Research goal: {goal}\n\n" if goal else ""
  ) + f"Webpage content:\n\n{snippet}"
  # A reasoning model can spend the whole output budget before it writes. Read
  # once more with a raised budget, and keep the last reply that has text.
  raw = ""
  error = None
  for max_tokens in _token_budgets(config):
    try:
      resp = await complete(
          profile,
          [
              {"role": "system", "content": EXTRACTOR_PROMPT},
              {"role": "user", "content": user},
          ],
          max_tokens=max_tokens,
          client=client,
      )
      choice = resp.choices[0]
      reply = choice.message.content or ""
    except Exception as exc:  # noqa: BLE001
      error = str(exc) or type(exc).__name__
      break
    if reply.strip():
      raw = reply
    if getattr(choice, "finish_reason", None) != "length":
      error = None
      break
    error = (
        f"the reader used all {max_tokens} output tokens before writing its"
        " summary"
    )
  if error and not raw.strip():
    # the reader failed; the page itself may be fine
    return {
        "accessible": False,
        "summary": "",
        "evidence": "",
        "reader_error": error,
    }

  obj = extract_json_obj(raw)
  if obj is None:
    if '"summary"' in raw or raw.lstrip().startswith("{"):
      salvaged = _salvage_reader_fields(raw)
      if salvaged["summary"]:
        return salvaged
      return {"accessible": False, "summary": "", "evidence": ""}
    return {
        "accessible": bool(raw.strip()),
        "summary": raw.strip(),
        "evidence": "",
    }
  return {
      "accessible": bool(obj.get("accessible", True)),
      "summary": str(obj.get("summary", "")).strip(),
      "evidence": str(obj.get("evidence", "")).strip(),
  }
