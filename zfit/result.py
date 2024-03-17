#  Copyright (c) 2024 zfit
from __future__ import annotations

from .minimizers.errors import compute_errors
from .minimizers.fitresult import Approximations, FitResult

__all__ = ["FitResult", "compute_errors", "Approximations"]
