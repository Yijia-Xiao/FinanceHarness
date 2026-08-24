"""The MCP ↔ harness translation layer — schemas in, results out.

Two directions share these primitives:

  inbound  (``hub``)    an external server's tool becomes a harness
                        :class:`ToolSpec`, so the agent loop calls it exactly
                        like a first-party tool (``prev:`` chaining included).
  outbound (``server``) a harness ToolSpec becomes an MCP tool, so any MCP
                        client can call it.

The schema seam is deliberately thin. An MCP tool advertises a JSON Schema, not
a Pydantic model, so we wrap it in a passthrough model that *reports* that schema
verbatim and accepts whatever the model sends. Validation then belongs to the
server that owns the contract (it answers with a structured error the model
self-corrects from) rather than to a lossy schema re-derivation here.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

from financeharness.runtime.tool_registry import ToolError

# Keys some strict function-calling validators reject; harmless to drop, since
# they constrain rather than describe the arguments.
_STRIPPED_SCHEMA_KEYS = ("$schema", "additionalProperties")

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def sanitize_json_schema(schema):
  """An MCP input schema made safe to hand to a function-calling API.

  Recursively drops the keys strict validators reject and normalizes a missing
  or non-object schema to an empty object schema (a no-argument tool).
  """
  if not isinstance(schema, dict) or not schema:
    return dict(_EMPTY_SCHEMA)

  def walk(node):
    if isinstance(node, dict):
      return {k: walk(v) for k, v in node.items() if k not in _STRIPPED_SCHEMA_KEYS}
    if isinstance(node, list):
      return [walk(v) for v in node]
    return node

  out = walk(schema)
  out.setdefault("type", "object")
  out.setdefault("properties", {})
  return out


def passthrough_model(model_name, json_schema):
  """A Pydantic model that *reports* ``json_schema`` and accepts any object.

  The harness registry is typed in Pydantic models, but an MCP tool's contract
  arrives as JSON Schema. Rather than re-derive an imperfect model from it (and
  reject valid calls the server would have accepted), the returned model passes
  arguments through and hands the schema straight to the model, unaltered.
  """
  schema = sanitize_json_schema(json_schema)
  model = create_model(  # extra="allow": every advertised field arrives as an extra
      model_name,
      __config__=ConfigDict(extra="allow"),
  )

  def model_json_schema(cls, **_kwargs):  # noqa: ARG001 — signature-compatible override
    return dict(schema)

  model.model_json_schema = classmethod(model_json_schema)
  return model


def call_arguments(validated):
  """The argument dict to send over the wire for a validated passthrough model."""
  if isinstance(validated, BaseModel):
    return validated.model_dump(mode="json", exclude_none=True)
  return dict(validated or {})


def _resource_block(resource):
  """One embedded resource rendered as markdown, text if it has any."""
  uri = str(getattr(resource, "uri", "") or "")
  text = getattr(resource, "text", None)
  if text:
    return f"**{uri}**\n\n{text}" if uri else str(text)
  mime = getattr(resource, "mime_type", None) or "application/octet-stream"
  return f"**{uri}** — binary resource ({mime}); not inlined."


def render_content(blocks):
  """MCP content blocks → one markdown string for the model.

  Text and embedded text resources inline; binary payloads (images, audio,
  blobs) are named rather than inlined — this loop reasons over text, and a
  base64 blob would only burn context.
  """
  parts: list[str] = []
  for block in blocks or []:
    kind = getattr(block, "type", None)
    if kind == "text":
      parts.append(str(getattr(block, "text", "") or ""))
    elif kind == "resource":
      parts.append(_resource_block(getattr(block, "resource", None)))
    elif kind == "resource_link":
      uri = getattr(block, "uri", "")
      desc = getattr(block, "description", None) or getattr(block, "name", "")
      parts.append(f"Resource link: {uri}" + (f" — {desc}" if desc else ""))
    elif kind in {"image", "audio"}:
      mime = getattr(block, "mime_type", None) or kind
      parts.append(f"({kind} payload, {mime} — not inlined)")
    else:  # an unknown future block type: name it rather than drop it silently
      parts.append(f"(unsupported content block: {kind})")
  return "\n\n".join(p for p in parts if p.strip())


def structured_payload(result, markdown):
  """The chaining surface for an MCP result.

  Prefers the server's own ``structuredContent``; falls back to parsing the text
  as JSON, then to the raw text. Something is always recorded, so a later call
  can reference this one with ``prev:<call_id>.<path>`` the same way it would a
  first-party tool.
  """
  structured = getattr(result, "structured_content", None)
  if isinstance(structured, dict):
    return structured
  text = markdown.strip()
  if text.startswith(("{", "[")):
    try:
      parsed = json.loads(text)
    except ValueError:
      parsed = None
    if isinstance(parsed, dict):
      return parsed
    if isinstance(parsed, list):
      return {"items": parsed}
  return {"text": markdown} if markdown else {}


def result_to_response(result, *, tool_label):
  """An MCP ``CallToolResult`` → ``(markdown, structured)``.

  Raises :class:`ToolError` when the server flags an error or answers with
  nothing, so the harness dispatcher renders the clean, actionable ``ok=False``
  a first-party tool failure produces.
  """
  markdown = render_content(getattr(result, "content", None))
  if getattr(result, "is_error", False):
    raise ToolError(markdown or f"{tool_label} reported an error with no detail.")
  structured = structured_payload(result, markdown)
  if not markdown and structured:
    markdown = f"```json\n{json.dumps(structured, indent=2, ensure_ascii=False)}\n```"
  if not markdown:
    raise ToolError(f"{tool_label} returned no content.")
  return markdown, structured


_CONTAINER_ORIGINS = (list, dict, set, tuple, frozenset)


def is_container_annotation(annotation):
  """True when a field holds a container (possibly inside a union/Optional).

  Marks the fields where reference chaining earns its keep: a price series or an
  FCF schedule is exactly what you don't want an LLM retyping.
  """
  if annotation in (None, Any):
    return False
  origin = get_origin(annotation)
  if origin in _CONTAINER_ORIGINS:
    return True
  if origin is not None:  # a union / Optional / Annotated wrapper: look inside
    return any(is_container_annotation(arg) for arg in get_args(annotation))
  return isinstance(annotation, type) and issubclass(annotation, _CONTAINER_ORIGINS)


def annotated_signature(request_schema):
  """A Pydantic request model → ``(parameter names, annotations, defaults)``.

  Used outbound: the MCP SDK derives a tool's schema from a Python signature, so
  a harness ToolSpec is exposed by synthesizing a function whose parameters
  mirror the model's fields. ``Annotated[type, FieldInfo]`` carries the field
  descriptions through, and required fields are ordered first so the signature is
  legal.

  Container fields are widened to ``T | str`` on purpose. The MCP layer validates
  arguments before the harness dispatcher resolves ``prev:<call_id>.<path>``
  references, so a strict ``list[float]`` would reject the very reference the
  harness exists to accept. The widened schema is the honest one: that field
  really does take either the values or a reference to them. Scalars stay strict
  — a single number is cheap to pass by value.
  """
  fields = list(request_schema.model_fields.items())
  required = [(n, f) for n, f in fields if f.is_required()]
  optional = [(n, f) for n, f in fields if not f.is_required()]
  ordered = required + optional
  annotations = {}
  for name, field in ordered:
    annotation = field.annotation
    if is_container_annotation(annotation):
      annotation = annotation | str
    annotations[name] = Annotated[annotation, field]
  defaults = tuple(field.get_default(call_default_factory=True) for _, field in optional)
  return [name for name, _ in ordered], annotations, defaults
