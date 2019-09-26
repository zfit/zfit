#  Copyright (c) 2019 zfit
import numpy as np
import pytest

import zfit
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.core.integration import binned_rect_integration
from zfit import z

low1 = (-1., 5., 3.)
high1 = (4.5, 9., 6.3)
space1 = zfit.Space(obs=[f'obs{i}' for i in range(3)], limits=((low1,), (high1,)))

size1 = (10000, 3)
test_data1 = np.random.uniform(low=low1, high=high1, size=size1)
n_bins1 = (6, 15, 11)
bincounts1, edges1 = np.histogramdd(test_data1, bins=n_bins1)
edges1 = np.array(edges1)


def test_binned_rect_integration():
    bincounts = z.convert_to_tensor(bincounts1)
    space = space1.with_autofill_axes(overwrite=True)
    integral = binned_rect_integration(bincounts, edges1, limits=space)
    subspace = space.get_subspace(obs=['obs0', 'obs2'])
    part_integral = binned_rect_integration(bincounts, edges1, limits=subspace)
    assert zfit.run(integral) == pytest.approx(size1[0])
    part_integral_np = zfit.run(part_integral)
    assert sum(part_integral_np) == pytest.approx(size1[0])
    assert part_integral.shape[0] == n_bins1[1]
