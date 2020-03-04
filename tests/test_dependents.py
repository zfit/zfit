#  Copyright (c) 2020 zfit

import zfit

# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


def test_get_dependents_is_deterministic():
    parameters = [zfit.Parameter(f'param{i}', i) for i in range(4)]
    obs = zfit.Space('obs1', (-3, 2))

    def create_pdf(params):
        param1, param2, param3, param4 = params
        gauss1 = zfit.pdf.Gauss(param1, param2, obs=obs)
        gauss2 = zfit.pdf.Gauss(param2, param2, obs=obs)
        gauss3 = zfit.pdf.Gauss(param2, param4, obs=obs)
        gauss4 = zfit.pdf.Gauss(param3, param1, obs=obs)
        sum_pdf = zfit.pdf.SumPDF([gauss1, gauss2, gauss3, gauss4], fracs=[0.1, 0.3, 0.4])
        return sum_pdf

    for params in (parameters, reversed(parameters)):
        pdf = create_pdf(params)
        assert pdf.get_dependents() == pdf.get_dependents(), "get_dependents() not deterministic"
