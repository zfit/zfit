#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from .minimizers.errors import compute_errors
from .minimizers.fitresult import Approximations, FitResult

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["Approximations", "FitResult", "compute_errors"]
