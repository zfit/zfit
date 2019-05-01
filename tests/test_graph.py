#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester


import sys

import zfit
from zfit import ztf
from zfit.util.graph import get_dependents_auto
from zfit.core.testing import setup_function, teardown_function, tester


# sys.setrecursionlimit(200)


def test_get_dependents():
    var1 = zfit.Parameter('var1', 1.)
    var2 = zfit.Parameter('var2', 2.)
    var3 = zfit.Parameter('var3', 3.)
    a = zfit.pdf.Gauss(var1, var2, obs='obs1').sample(n=500, limits=(-5, 5)) * 5.
    b = ztf.constant(3.) + 4 * var1
    c = 5. * b * var3
    d = b * var2 + a
    e = d * 3.
    zfit.run(e)
    assert get_dependents_auto(e, [a, b, c, d, var1, var2, var3]) == [a, b, d, var1, var2]
    assert get_dependents_auto(e, [var1, var2, var3]) == [var1, var2]
    assert get_dependents_auto(c, [a, b, d, var1, var2, var3]) == [b, var1, var3]
