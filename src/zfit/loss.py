#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from ._interfaces import ZfitLoss
from ._loss.binnedloss import (
    BinnedChi2,
    BinnedNLL,
    ExtendedBinnedChi2,
    ExtendedBinnedNLL,
)
from ._loss.general import Chi2
from .core.loss import BaseLoss, ExtendedUnbinnedNLL, SimpleLoss, UnbinnedNLL

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

# Set __module__ to zfit.loss for proper documentation linking
# Classes from zfit.core.loss
UnbinnedNLL.__module__ = "zfit.loss"
ExtendedUnbinnedNLL.__module__ = "zfit.loss"
BaseLoss.__module__ = "zfit.loss"
SimpleLoss.__module__ = "zfit.loss"
# Classes from zfit._loss.binnedloss
BinnedNLL.__module__ = "zfit.loss"
ExtendedBinnedNLL.__module__ = "zfit.loss"
BinnedChi2.__module__ = "zfit.loss"
ExtendedBinnedChi2.__module__ = "zfit.loss"
# Interface from zfit._interfaces
ZfitLoss.__module__ = "zfit.loss"

__all__ = [
    "BaseLoss",
    "BinnedChi2",
    "BinnedNLL",
    "Chi2",
    "ExtendedBinnedChi2",
    "ExtendedBinnedNLL",
    "ExtendedUnbinnedNLL",
    "SimpleLoss",
    "UnbinnedNLL",
    "ZfitLoss",
]
