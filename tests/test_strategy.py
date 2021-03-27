#  Copyright (c) 2021 zfit
import math

import pytest

import zfit


@pytest.mark.flaky(2)
def test_fail_on_nan_strategy():
    from zfit.minimizers.strategy import ToyStrategyFail
    sigma = zfit.Parameter("sigma", 2.)
    obs = zfit.Space("obs1", limits=(-4, 5))
    gauss = zfit.pdf.Gauss(1., sigma, obs=obs)

    sampler = gauss.create_sampler(3000)
    sampler.set_data_range(obs)
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=sampler)
    minimizer = zfit.minimize.Minuit(strategy=ToyStrategyFail)
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


def minimizers():
    return [
        zfit.minimize.Adam,
        zfit.minimize.NLoptMMAV1,
        zfit.minimize.IpyoptV1,
        zfit.minimize.Minuit,
        zfit.minimize.ScipySLSQPV1,
    ]


# sort for xdist: https://github.com/pytest-dev/pytest-xdist/issues/432
minimizers = sorted(minimizers(), key=lambda val: repr(val))


@pytest.mark.parametrize('minimizer_cls', minimizers)
def test_callback(minimizer_cls):
    from zfit.minimizers.strategy import PushbackStrategy

    class MyError(Exception):
        pass

    class MyStrategy(PushbackStrategy):
        def callback(*args, **kwargs):
            raise MyError

    minimizer = zfit.minimize.ScipySLSQPV1(strategy=MyStrategy)
    loss = lambda *args, **kwargs: 42.
    loss.errordef = 0.5
    with pytest.raises(MyError):
        minimizer.minimize(loss, [1, 3])
