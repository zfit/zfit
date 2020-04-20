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
        assert pdf.get_cache_deps() == pdf.get_cache_deps(), "get_dependents() not deterministic"


def test_get_params():
    obs = zfit.Space('obs', (-4, 5))
    mu = zfit.Parameter('mu', 1)
    mu2 = zfit.Parameter('mu2', 1)
    sigma2 = zfit.Parameter('sigma2', 2)
    sigma = zfit.ComposedParameter('sigma', lambda s: s * 0.7, params=sigma2)

    yield1 = zfit.Parameter('yield1', 10)
    yield2free = zfit.Parameter('yield2free', 200)
    yield2 = zfit.ComposedParameter('yield2', lambda y: y * 0.9, params=yield2free)

    gauss = zfit.pdf.Gauss(mu, sigma, obs)
    gauss2 = zfit.pdf.Gauss(mu2, sigma2, obs)
    gauss_ext = gauss.create_extended(yield1)
    gauss2_ext = gauss2.create_extended(yield2)

    assert set(gauss.get_params()) == set([mu, sigma2])
    assert set(gauss_ext.get_params(yields=False)) == set([mu, sigma2])
    assert set(gauss_ext.get_params(yields=False, floating=None, extract_independent=None)) == set([mu, sigma])
    assert set(gauss_ext.get_params(yields=True, floating=None, extract_independent=None)) == set([mu, sigma, yield1])
    assert set(gauss_ext.get_params()) == set([mu, sigma2, yield1])
