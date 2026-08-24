"""The `calc` tool — exact arithmetic so the model never computes in its head.

A safe AST evaluator over numbers, the standard operators, and a small whitelist
of functions. LLMs are reliable at reasoning and unreliable at arithmetic (a
recurring faithfulness failure: "$37B is a 123% increase from $13B"); offloading
every growth rate, ratio, and percentage to this tool keeps the numbers exact
and
auditable.
"""

from __future__ import annotations

import ast
import math
import operator

from pydantic import BaseModel, Field

from financeharness.runtime.tool_registry import ToolError, ToolResponse, ToolSpec

_DESCRIPTION = (
    "Evaluate an arithmetic expression exactly and return the result — use it"
    " for growth rates, ratios, percentages, and sums so the figures are exact"
    " and auditable, e.g. '(37-13)/13*100' for a percent change. Supports + - *"
    " / ** % // , parentheses, and abs/round/min/max/sqrt/log/exp."
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# Bound `**` so a model-emitted `10**10**10` can't build a multi-billion-digit
# integer and block the event loop (tool handlers run inline in the async loop).
_MAX_EXPONENT = 1000
# A per-node exponent cap is not sufficient on its own, because exponents
# compose: `((10**1000)**1000)**10` keeps every individual exponent inside the
# cap while the value grows multiplicatively. Bound the size of the *result* as
# well, so a chain is refused at whichever step first exceeds the budget.
# 16384 bits is ~4900 digits, comfortably above anything financial arithmetic
# needs and above what the exponent cap alone already allowed (10**1000).
_MAX_RESULT_BITS = 16384
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}


def _check_pow(left, right):
  """Refuse a power whose operands or result would be unreasonably large.

  Integer exponentiation is the one operator here that can turn a short
  expression into an enormous amount of work, and it is the result size rather
  than the literal exponent that determines the cost. For integers the result
  needs about ``bit_length(base) * exponent`` bits, which is cheap to check
  before doing any of the work. Floats need no size check: an overflowing float
  power raises OverflowError, which ``safe_eval`` already turns into a
  ToolError.
  """
  if abs(right) > _MAX_EXPONENT:
    raise ToolError(
        f"exponent too large (|exponent| > {_MAX_EXPONENT}) — refused to"
        " avoid a runaway computation"
    )
  if (
      isinstance(left, int)
      and isinstance(right, int)
      and right > 0
      and max(left.bit_length(), 1) * right > _MAX_RESULT_BITS
  ):
    raise ToolError(
        f"result too large (would exceed {_MAX_RESULT_BITS} bits) — refused to"
        " avoid a runaway computation"
    )


def _eval(node):
  """Recursively evaluate a parsed expression, allowing only safe nodes."""
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return node.value
  if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
    left, right = _eval(node.left), _eval(node.right)
    if type(node.op) is ast.Pow:
      _check_pow(left, right)
    return _BIN_OPS[type(node.op)](left, right)
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
    return _UNARY_OPS[type(node.op)](_eval(node.operand))
  if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id in _FUNCS
      and not node.keywords
  ):
    return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
  raise ToolError(
      "unsupported expression — numbers, + - * / ** % //, () and "
      "abs/round/min/max/sqrt/log/exp only"
  )


def safe_eval(expression):
  """Evaluate an arithmetic expression safely (no names, calls, or attribute access

  beyond the whitelisted functions). Raises ToolError on a malformed or
  unsupported expression (a parse error, a bad domain like sqrt(-1), or division
  by zero) so the model gets a clean, actionable message.
  """
  try:
    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)
  except ToolError:
    raise
  except Exception as exc:  # noqa: BLE001 — parse/arithmetic errors → clean ToolError
    raise ToolError(f"could not evaluate '{expression}': {exc}") from exc


class CalcRequest(BaseModel):
  """Input for the exact arithmetic tool."""

  expression: str = Field(
      Ellipsis, description="Arithmetic expression, e.g. '(37-13)/13*100'."
  )


async def _handler(req):
  result = safe_eval(req.expression)
  # tidy float display without lying about precision
  try:
    shown = f"{result:.6g}"
  except OverflowError:
    # an exact integer too large to convert to float (e.g. 10**400, well inside
    # the exponent limit) — show it exactly rather than failing the call
    shown = f"{result:d}"
  return ToolResponse(
      markdown=f"`{req.expression}` = **{shown}**",
      structured={"expression": req.expression, "result": result},
      meta={},
  )


SPEC = ToolSpec(
    name="calc",
    display_name="calc",
    tier="core",  # always visible — arithmetic comes up in every mode
    description=_DESCRIPTION,
    request_schema=CalcRequest,
    handler=_handler,
    tags=("compute", "arithmetic"),
)
