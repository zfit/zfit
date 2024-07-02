#  Copyright (c) 2024 zfit
from __future__ import annotations

from ._data.binneddatav1 import BinnedData, BinnedSamplerData
from ._variables.axis import RegularBinning, VariableBinning
from .core.data import Data, SamplerData, concat, convert_to_data

__all__ = [
    "Data",
    "BinnedData",
    "RegularBinning",
    "VariableBinning",
    "convert_to_data",
    "SamplerData",
    "BinnedSamplerData",
    "concat",
    "from_numpy",
    "from_pandas",
    "from_root",
    "from_binned_tensor",
    "from_hist",
]

# create aliases for class constructors
from_numpy = Data.from_numpy
from_pandas = Data.from_pandas
from_root = Data.from_root
from_binned_tensor = BinnedData.from_tensor
from_hist = BinnedData.from_hist
