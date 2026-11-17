"""Shared numeric field types for the compute tools.

``json.loads`` accepts ``Infinity``, ``NaN`` and overflowing literals such as
``1e400``, and Pydantic floats admit them by default. ``Finite`` rejects them at
validation. Fields with an upper bound need it only for clarity: ``lt``/``le``
excludes ``inf`` and any bound excludes ``nan``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

Finite = Annotated[float, Field(allow_inf_nan=False)]
