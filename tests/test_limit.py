#  Copyright (c) 2020 zfit
import pytest

from zfit import z
from zfit.core.space_new import Limit
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.util.exception import IllegalInGraphModeError


@pytest.mark.parametrize('graph', [True, False])
def test_rect_limit(graph):
    rect_limits = (1., 3)

    def test(allow_graph=True):
        limit = Limit(rect_limits)
        limit2 = Limit(rect_limits=rect_limits)

        lower, upper = limit.rect_limits
        lower2, upper2 = limit2.rect_limits
        assert limit.has_rect_limits
        inside = limit.inside(2)
        inside2 = limit.inside(2, guarantee_limits=True)
        outside = limit.inside(4)
        if graph:
            equal = limit.equal(limit2, allow_graph=allow_graph)
        else:
            equal = limit == limit2

        return lower, upper, lower2, upper2, inside, inside2, outside, equal

    if graph:
        test = z.function(test)
    lower, upper, lower2, upper2, inside, inside2, outside, equal = test()
    assert equal
    assert lower, upper == rect_limits
    assert lower2, upper2 == rect_limits
    assert inside
    assert inside2
    assert not outside
