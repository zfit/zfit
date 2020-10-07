#  Copyright (c) 2020 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.models.dist_tfp import Poisson

lamb_true = 50

obs = zfit.Space(obs="Nobs", limits=(0, 200))

test_values = np.random.uniform(low=0, high=100, size=100)


def create_poisson():
    N = Parameter("N", lamb_true)
    poisson = Poisson(obs=obs, lamb=N)
    return poisson


def test_poisson():
    poisson = create_poisson()

    probs1 = poisson.pdf(x=test_values)
    probs1 = probs1.numpy()

    samples = poisson.sample(10000).numpy()

    assert np.std(samples) == pytest.approx(50**0.5, rel=0.05)
