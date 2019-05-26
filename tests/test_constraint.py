#  Copyright (c) 2019 zfit
import pytest
import numpy as np

import zfit
from zfit import ztf
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.exception import ShapeIncompatibleError


def test_shape_errors():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)

    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=[4, 2, 3], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=[4, 2], sigma=5)
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian([param1, param2], mu=2, sigma=[1, 4])
    with pytest.raises(ShapeIncompatibleError):
        zfit.constraint.nll_gaussian(param1, mu=[4, 2], sigma=[2, 3])


def test_nll_gaussian_values():
    param1 = zfit.Parameter("Param1", 5)
    param2 = zfit.Parameter("Param2", 6)
    params = [param1, param2]

    mu = [3., 6.1]
    sigma = np.array([[1, 0.3],
                      [0.8, 0.5]])
    constr = zfit.constraint.nll_gaussian(params=params, mu=mu, sigma=sigma)
    constr_np = zfit.run(constr)
    assert constr_np == pytest.approx(4.28846)
