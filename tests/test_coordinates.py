#  Copyright (c) 2020 zfit
import pytest
import tensorflow as tf
import numpy as np

from zfit import z
from zfit.core.space_new import Coordinates
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


@pytest.mark.parametrize('graph', [True, False])
def test_reorder_x(graph):
    x = tf.constant([2, 3, 4, 1], dtype=tf.float64)

    def test():
        coords = Coordinates(['a', 'b', 'c', 'd'])
        x1 = coords.reorder_x(x, x_obs=['b', 'c', 'd', 'a'])
        x2 = coords.reorder_x(x, func_obs=['d', 'a', 'b', 'c', ])

        return x1, x2

    if graph:
        test = z.function(test)
    x1, x2 = test()
    true_x = tf.constant([1, 2, 3, 4], dtype=tf.float64)
    assert np.allclose(x1, true_x)
    assert np.allclose(x2, true_x)
