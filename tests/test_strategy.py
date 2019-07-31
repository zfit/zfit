#  Copyright (c) 2019 zfit
from zfit.core.testing import teardown_function, setup_function

import math

import pytest

import zfit
from zfit.minimizers.baseminimizer import ToyStrategyFail


@pytest.mark.flaky(2)
def test_fail_on_nan_strategy():
    sigma = zfit.Parameter("sigma", 2.)
    obs = zfit.Space("obs1", limits=(-4, 5))
    gauss = zfit.pdf.Gauss(1., sigma, obs=obs)

    sampler = gauss.create_sampler(3000)
    sampler.set_data_range(obs)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=sampler)
    minimizer = zfit.minimize.Minuit(strategy=ToyStrategyFail())
    sampler.resample()
    sigma.set_value(2.1)
    fitresult1 = minimizer.minimize(nll)
    assert fitresult1.converged

    sampler.resample()
    sigma.set_value(math.inf)
    fitresult2 = minimizer.minimize(nll)
    assert not fitresult2.converged
    assert fitresult2.edm == -999
    assert fitresult2.fmin == -999
