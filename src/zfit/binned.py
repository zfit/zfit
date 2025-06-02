#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

from ._variables.axis import Binnings, RegularBinning, VariableBinning

__all__ = ["Binnings", "RegularBinning", "VariableBinning"]
