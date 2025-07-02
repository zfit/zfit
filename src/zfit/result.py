#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from ._minimizers.errors import compute_errors
from ._minimizers.fitresult import Approximations, FitResult

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["Approximations", "FitResult", "compute_errors"]
