#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from ._variables.axis import Binnings, RegularBinning, VariableBinning

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["Binnings", "RegularBinning", "VariableBinning"]
