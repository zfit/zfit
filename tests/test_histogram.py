#  Copyright (c) 2019 zfit
import numpy as np

import zfit
from zfit.core.testing import setup_function, teardown_function, tester


def test_histogramdd():
    data = np.random.normal(size=(1000, 3))

    hist = zfit.hist.histogramdd(sample=data)
    bincount_np, edges_np = zfit.run(hist)
