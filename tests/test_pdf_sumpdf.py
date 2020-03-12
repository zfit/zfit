#  Copyright (c) 2020 zfit
import numpy as np
import pytest

import zfit
from zfit import Parameter
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester
from zfit.models.dist_tfp import Gauss


def test_analytic_integral():
    obs = zfit.Space('obs1', (-2, 5))
