"""Shared numeric field types for the compute tools.

Tool arguments arrive as model-emitted JSON, and ``json.loads`` accepts both the
bare ``Infinity``/``NaN`` tokens and overflowing literals such as ``1e400``.
Pydantic's ``allow_inf_nan`` defaults to True, so without an explicit constraint
a non-finite value validates cleanly, flows through the arithmetic, and only
surfaces much later — as a dollar figure in the rendered markdown, or as a
serialization failure when the trajectory is written out, since ``inf`` and
``nan`` are not valid JSON. Rejecting at the schema boundary turns it into an
ordinary validation error the model can read and correct.

Only fields whose existing bounds don't already exclude non-finite values need
this: an upper bound (``lt``/``le``) rules out ``inf`` on its own, and every
comparison against ``nan`` is False, so any bound at all rules out ``nan``. A
lower bound alone (``ge=0``) still admits ``inf``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

Finite = Annotated[float, Field(allow_inf_nan=False)]
