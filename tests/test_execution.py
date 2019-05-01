#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester


import copy

import pytest

import zfit
from zfit import ztf
from zfit.core.testing import setup_function, teardown_function, tester


def test_run():
    a = ztf.constant(4.)
    b = 5 * a
    assert zfit.run(b) == pytest.approx(20)


@pytest.mark.parametrize(["n_cpu", 'taken', 'left'], [[3, 5, 0], [0, -1, 0], [10, 3, 7],
                                                      [5, -1, 0], [8, -3, 2], [5, 0, 5]])
def test_cpu_management(n_cpu, taken, left):
    zfit.run.set_n_cpu(n_cpu=n_cpu)
    _cpu = copy.deepcopy(zfit.run._cpu)
    assert zfit.run.n_cpu == n_cpu
    with zfit.run.aquire_cpu(max_cpu=taken) as cpus:
        assert zfit.run.n_cpu == left
        assert isinstance(cpus, list)
        assert len(cpus) == n_cpu - left
    assert zfit.run.n_cpu == n_cpu
    assert _cpu == zfit.run._cpu
