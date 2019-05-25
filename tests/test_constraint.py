#  Copyright (c) 2019 zfit
import pytest

import zfit
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.exception import ShapeIncompatibleError


def test_shape_errors():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 5)

    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=[4, 2, 3], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=[4, 2], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=2, sigma=[1, 4])
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian(param1, mu=[4, 2], sigma=[2, 3])
