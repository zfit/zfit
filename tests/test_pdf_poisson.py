#  Copyright (c) 2023 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
from zfit.models.dist_tfp import Poisson

lamb_true = 50

obs = zfit.Space(obs="Nobs", limits=(0, 200))

test_values = np.random.uniform(low=0, high=100, size=100)


def create_poisson():
    N = Parameter("N", lamb_true)
    poisson = Poisson(obs=obs, lam=N)
    return poisson


def create_poisson_composed_rate():
    N1 = Parameter("N1", lamb_true / 2)
    N2 = Parameter("N2", lamb_true / 2)
    N = zfit.param.ComposedParameter("N", lambda n1, n2: n1 + n2, params=[N1, N2])

    poisson = Poisson(obs=obs, lam=N)
    return poisson


@pytest.mark.parametrize("composed_rate", [False, True])
def test_poisson(composed_rate):
    if composed_rate:
        poisson = create_poisson_composed_rate()
    else:
        poisson = create_poisson()

    probs1 = poisson.pdf(x=test_values)
    probs1 = probs1.numpy()

    samples = poisson.sample(10000).numpy()

    assert np.std(samples) == pytest.approx(50**0.5, rel=0.05)
