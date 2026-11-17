"""JSON helpers for model output: object extraction, strict parsing, safe values."""

from __future__ import annotations

import json
import math
import numbers
import re


def _finite_float(text):
  value = float(text)
  if not math.isfinite(value):
    raise ValueError(f"number out of range: {text}")
  return value


def _reject_constant(name):
  raise ValueError(f"{name} is not a JSON value")


def loads_strict(text):
  """Parse JSON text, accepting only what strict JSON and UTF-8 can carry.

  Raises ValueError for ``NaN``, ``Infinity``, numbers that overflow to
  infinity (``1e400``) and unpaired surrogate escapes, all of which
  ``json.loads`` accepts.
  """
  value = json.loads(text, parse_float=_finite_float, parse_constant=_reject_constant)
  json.dumps(value, ensure_ascii=False).encode("utf-8")  # unpaired surrogates raise
  return value


def json_safe(value):
  """``value`` with numbers made JSON-safe, for a payload that will be serialized.

  Non-finite floats become None (a missing figure), numeric scalars of other
  libraries become int or float, tuples become lists and dict keys strings.
  Raises TypeError for anything else JSON cannot carry.
  """
  if value is None or isinstance(value, (bool, str)):
    return value
  if isinstance(value, numbers.Integral):
    return int(value)
  if isinstance(value, numbers.Real):
    number = float(value)
    return number if math.isfinite(number) else None
  if isinstance(value, dict):
    return {str(k): json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(v) for v in value]
  raise TypeError(f"{type(value).__name__} is not a JSON value")


def extract_json_obj(raw):
  """Parse a JSON object from model text, tolerating code fences / surrounding

  prose. Returns None if no object is found.
  """
  raw = (raw or "").strip()
  try:
    obj = json.loads(raw)
    return obj if isinstance(obj, dict) else None
  except (json.JSONDecodeError, ValueError):
    pass
  m = re.search(r"\{.*\}", raw, re.DOTALL)
  if not m:
    return None
  try:
    obj = json.loads(m.group(0))
    return obj if isinstance(obj, dict) else None
  except (json.JSONDecodeError, ValueError):
    return None
