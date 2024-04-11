#  Copyright (c) 2024 zfit
from __future__ import annotations

from .core.coordinates import Coordinates
from .core.space import Space, combine_spaces

__all__ = [
    "Space",
    "combine_spaces",
    "Coordinates",
]
