#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from .core.coordinates import Coordinates
from .core.space import Space, combine_spaces

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = [
    "Coordinates",
    "Space",
    "combine_spaces",
]
