#  Copyright (c) 2020 zfit
import pytest
import tensorflow as tf

import zfit
from zfit import z
from zfit.core.space import Limit, ANY
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester

rect_limits = (1., 3)
rect_limits_tf = (tf.constant(1.), tf.constant(3))
rect_limit_enlarged = rect_limits[0] - 1, rect_limits[1] + 1
rect_limits_any = (ANY, ANY)


def inside_rect_limits(x):
    return tf.logical_and(rect_limits[0] < x, x < rect_limits[1])[:, 0]


@pytest.mark.parametrize('graph', [True, False])
@pytest.mark.parametrize('rect_limits', [rect_limits, rect_limits_tf])
@pytest.mark.parametrize('testclass',
                         [Limit, lambda limit_fn=None, rect_limits=None: zfit.Space('obs1', limits=limit_fn,
                                                                                    rect_limits=rect_limits)])
def test_rect_limits_1d(graph, rect_limits, testclass):
    def test(allow_graph=True):
        limit = testclass(rect_limits)
        limit2 = testclass(rect_limits=rect_limits)

        lower, upper = limit.rect_limits
        lower2, upper2 = limit2.rect_limits
        assert limit.has_rect_limits
        inside = limit.inside(2)
        inside2 = limit.inside(2, guarantee_limits=True)
        outside = limit.inside(4)
        if graph:
            equal = limit.equal(limit2, allow_graph=allow_graph)
            less_equal = limit.less_equal(limit2, allow_graph=allow_graph)
        else:
            equal = limit == limit2
            less_equal = limit <= limit2

        return lower, upper, lower2, upper2, inside, inside2, outside, equal, less_equal

    if graph:
        test = z.function(test)
    lower, upper, lower2, upper2, inside, inside2, outside, equal, less_equal = test()
    assert equal
    assert less_equal
    assert lower, upper == rect_limits
    assert lower2, upper2 == rect_limits
    assert inside
    assert inside2
    assert not outside


@pytest.mark.parametrize('graph', [True, False])
@pytest.mark.parametrize('testclass', [Limit, lambda limit_fn, rect_limits: zfit.Space('obs1', limits=limit_fn,
                                                                                       rect_limits=rect_limits)])
def test_less_equal(graph, testclass):
    def test(allow_graph=True):
        limit = testclass(limit_fn=inside_rect_limits, rect_limits=rect_limits)
        limit2 = testclass(limit_fn=inside_rect_limits, rect_limits=rect_limit_enlarged)

        assert not limit.has_rect_limits
        assert limit.has_limits
        assert not limit.limits_not_set
        assert not limit.limits_are_false
        inside = limit.inside(2)
        inside2 = limit.inside(2, guarantee_limits=True)
        outside = limit.inside(4)

        assert not limit2.has_rect_limits
        assert limit2.has_limits
        assert not limit2.limits_not_set
        assert not limit2.limits_are_false
        inside21 = limit2.inside(2)
        inside22 = limit2.inside(2, guarantee_limits=True)
        outside2 = limit2.inside(4)
        if graph:
            equal = limit.equal(limit2, allow_graph=allow_graph)
        else:
            equal = limit == limit2

        return inside, inside2, outside, equal, inside21, inside22, outside2

    if graph:
        test = z.function(test)
    inside, inside2, outside, equal, inside21, inside22, outside2 = test()
    assert not equal
    assert inside
    assert inside2
    assert not outside
    assert inside21
    assert inside22
    assert not outside2


@pytest.mark.parametrize('graph', [True, False])
@pytest.mark.parametrize('testclass', [Limit, lambda limit_fn, rect_limits: zfit.Space('obs1', limits=limit_fn,
                                                                                       rect_limits=rect_limits)])
def test_limits_1d(graph, testclass):
    def test(allow_graph=True):
        limit = testclass(limit_fn=inside_rect_limits, rect_limits=rect_limits)
        limit2 = testclass(limit_fn=inside_rect_limits, rect_limits=rect_limit_enlarged)

        assert not limit.has_rect_limits
        assert limit.has_limits
        assert not limit.limits_not_set
        assert not limit.limits_are_false
        inside = limit.inside(2)
        inside2 = limit.inside(2, guarantee_limits=True)
        outside = limit.inside(4)

        assert not limit2.has_rect_limits
        assert limit2.has_limits
        assert not limit2.limits_not_set
        assert not limit2.limits_are_false
        inside21 = limit2.inside(2)
        inside22 = limit2.inside(2, guarantee_limits=True)
        outside2 = limit2.inside(4)
        if graph:
            equal = limit.equal(limit2, allow_graph=allow_graph)
        else:
            equal = limit == limit2

        return inside, inside2, outside, equal, inside21, inside22, outside2

    if graph:
        test = z.function(test)
    inside, inside2, outside, equal, inside21, inside22, outside2 = test()
    assert not equal
    assert inside
    assert inside2
    assert not outside
    assert inside21
    assert inside22
    assert not outside2
