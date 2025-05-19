#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

from .core.coordinates import Coordinates
from .core.space import Space, combine_spaces

__all__ = [
    "Coordinates",
    "Space",
    "combine_spaces",
]
