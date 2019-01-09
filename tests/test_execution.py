import pytest

import zfit
from zfit import ztf


def test_run():
    a = ztf.constant(4.)
    b = 5 * a
    assert zfit.run(b) == pytest.approx(20)
