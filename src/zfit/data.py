#  Copyright (c) 2025 zfit
from __future__ import annotations

import typing

from ._data.binneddatav1 import BinnedData, BinnedSamplerData
from ._variables.axis import RegularBinning, VariableBinning
from .core.data import Data, SamplerData, concat, convert_to_data

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = [
    "BinnedData",
    "BinnedSamplerData",
    "Data",
    "RegularBinning",
    "SamplerData",
    "VariableBinning",
    "concat",
    "convert_to_data",
    "from_binned_tensor",
    "from_hist",
    "from_numpy",
    "from_pandas",
    "from_root",
]

# create aliases for class constructors
from_numpy = Data.from_numpy
from_pandas = Data.from_pandas
from_root = Data.from_root
from_binned_tensor = BinnedData.from_tensor
from_hist = BinnedData.from_hist
