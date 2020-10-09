#  Copyright (c) 2020 zfit
from typing import List

import boost_histogram as bh
import numpy as np

from zfit.core.interfaces import ZfitRectBinning


class RectBinning(ZfitRectBinning):

    def __init__(self, binnings):
        super().__init__()
        self._binnings = binnings

    def get_binning(self) -> List[bh.axis.Axis]:
        return self._binnings

    def get_edges(self) -> np.array:
        # TODO: create from axes
        pass
