#  Copyright (c) 2024 zfit
import numpy as np
import pytest
from pytest_cases import parametrize, fixture

import zfit
import zfit.z.numpy as znp


# todo: expand tests with pdfs?

def pdf_gauss():
    scale = zfit.Parameter('scale1', 3065.)
    mu = zfit.Parameter('muparam', 1.2, -1., 3.)
    sigma = zfit.Parameter('sigmaparam', 13, 0.1, 100.)
    obs = zfit.Space('obs1', -10, 10)

    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=scale)
    return gauss


def pdf_gaussnonext():
    mu = zfit.Parameter('muparam', 1.2, -1., 3.)
    sigma = zfit.Parameter('sigmaparam', 1.3, 0.1, 10.)
    obs = zfit.Space('obs1', -10, 10)

    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    return gauss


def pdf_expnonext():
    lambda_ = zfit.Parameter('lambda', -0.03, -2., 0.)
    obs = zfit.Space('obs1', -10, 10)

    exp = zfit.pdf.Exponential(lambda_=lambda_, obs=obs)
    return exp


def pdf_sumnonext():
    sum_pdf = zfit.pdf.SumPDF(pdfs=[pdf_gaussnonext(), pdf_expnonext()], fracs=zfit.Parameter('frac', 0.25))
    return sum_pdf


def pdf_exp():
    lambda_ = zfit.Parameter('lambda', -0.03, -2., 0.)
    scale = zfit.Parameter('scale2', 10001.)
    obs = zfit.Space('obs1', -10, 10)

    exp = zfit.pdf.Exponential(lambda_=lambda_, obs=obs, extended=scale)
    return exp


def pdf_cheby():
    c0 = zfit.Parameter('c0', 1.)
    c1 = zfit.Parameter('c1', 0.5)
    obs = zfit.Space('obs1', -10, 10)
    scale = zfit.Parameter('scale3', 10101.)

    cheby = zfit.pdf.Chebyshev(obs=obs, coeffs=[c0, c1], extended=scale)
    return cheby


def pdf_prod():
    prod = zfit.pdf.ProductPDF(pdfs=[pdf_gauss(), pdf_cheby()], extended=pdf_gauss().get_yield())
    return prod


def pdf_sum():
    sum_pdf = zfit.pdf.SumPDF(pdfs=[pdf_prod(), pdf_exp()])
    return sum_pdf


# @parametrize('pdf', [pdf_gauss, pdf_exp, pdf_cheby, pdf_prod, pdf_sum])
@pytest.mark.parametrize('binning', [None, 113])
def test_pdf_params_args_methods(binning):
    mu = zfit.Parameter('muparam', 1.2)
    sigma = zfit.Parameter('sigmaparam', 1.1)
    scale = zfit.Parameter('scale', 10.)
    obs = zfit.Space('obs1', -10, 10, binning=binning)

    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, extended=scale)
    if binning is not None:
        gauss = gauss.to_binned(obs)

    data = znp.array([1.1, 1.2, 1.3, 1.4, 1.5])

    assert np.argmax(gauss.pdf(data)) == 1
    probs = gauss.pdf(data, params={'muparam': 1.4})
    assert np.argmax(probs) == 3
    with pytest.raises(ValueError):
        gauss.pdf(data, params={'muparam': 1.4, 'sigma': 1.5})
    assert np.argmax(gauss.pdf(data)) == 1

    assert np.argmax(gauss.ext_pdf(data, params={'muparam': 1.4, "sigmaparam": 1.1})) == 3
    with pytest.raises(ValueError):
        gauss.ext_pdf(data, params={'mupar': 1.4, 'sigmaparam': 1.5})
    assert np.argmax(gauss.ext_pdf(data)) == 1

    assert np.argmax(gauss.log_pdf(data, params={'muparam': 1.4, "sigmaparam": 1.1})) == 3
    with pytest.raises(ValueError):
        gauss.log_pdf(data, params={'mupar': 1.4, 'sigmaparam': 1.5})
    assert np.argmax(gauss.log_pdf(data)) == 1

    assert np.argmax(gauss.ext_log_pdf(data, params={'muparam': 1.4, "sigmaparam": 1.1})) == 3
    with pytest.raises(ValueError):
        gauss.ext_log_pdf(data, params={'mupar': 1.4, 'sigmaparam': 1.5})

    assert np.argmax(gauss.ext_log_pdf(data)) == 1

    integral_diff = gauss.integrate(limits=(-1, 2)) - gauss.integrate(limits=(-1, 2), params={'sigmaparam': 1.4})
    assert integral_diff > 0
    with pytest.raises(ValueError):
        gauss.integrate(limits=(-1, 2), params={'sigmaparam': 1.4, 'mu': 1.5})

    integral_diff = gauss.ext_integrate(limits=(-1, 2)) - gauss.ext_integrate(limits=(-1, 2), params={'sigmaparam': 1.4})
    assert integral_diff > 0
    with pytest.raises(ValueError):
        gauss.ext_integrate(limits=(-1, 2), params={'sigmaparam': 1.4, 'mu': 1.5})

    if binning is not None:
        assert np.argmax(gauss.counts(data)) == 1
        assert np.argmax(gauss.counts(data, params={'muparam': 1.4})) == 3
        with pytest.raises(ValueError):
            gauss.counts(data, params={'muparam': 1.4, 'sigma': 1.5})
        assert np.argmax(gauss.counts(data)) == 1

        assert np.argmax(gauss.rel_counts(data, params={'muparam': 1.4, "sigmaparam": 1.1})) == 3
        with pytest.raises(ValueError):
            gauss.rel_counts(data, params={'mupar': 1.4, 'sigmaparam': 1.5})
        assert np.argmax(gauss.rel_counts(data)) == 1


def pdf_binned():
    return pdf_sum().to_binned(142)


def unbinnednll():
    pdf = pdf_sumnonext()
    data = pdf.create_sampler(n=10000)
    loss = zfit.loss.UnbinnedNLL(model=pdf, data=data)
    return loss


def ext_unbinnednll():
    pdf = pdf_sum()
    data = pdf.create_sampler(n=10000)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=pdf, data=data)
    return loss


def binnednll():
    pdf = pdf_binned()
    databinned = pdf.sample(n=10000)
    loss = zfit.loss.BinnedNLL(model=pdf, data=databinned)
    return loss


def ext_binnednll():
    pdf = pdf_binned()
    databinned = pdf.sample(n=10000)
    loss = zfit.loss.ExtendedBinnedNLL(model=pdf, data=databinned)
    return loss


def chi2():
    pdf = pdf_binned()
    databinned = pdf.sample(n=10000)
    databinned = databinned.with_variances(databinned.counts())
    loss = zfit.loss.BinnedChi2(model=pdf, data=databinned)
    return loss


def ext_chi2():
    pdf = pdf_binned()
    databinned = pdf.sample(n=10000)
    databinned = databinned.with_variances(databinned.counts())
    loss = zfit.loss.ExtendedBinnedChi2(model=pdf, data=databinned)
    return loss


losses = [unbinnednll, ext_unbinnednll, binnednll, ext_binnednll, chi2, ext_chi2]


@pytest.mark.parametrize('loss', losses)
def test_params_as_args_loss(loss):
    loss = loss()
    params = loss.get_params()
    param1 = list(params)[0]
    loss_values = loss.value()
    assert np.isfinite(loss_values)
    loss_values_params = loss.value(params={param1.name: param1 * 1.1})
    assert abs(loss_values - loss_values_params) > 0.1

@pytest.mark.parametrize('loss', losses)
@pytest.mark.flaky(reruns=2)  # minimization not reliable
def test_params_as_args_fitresult(loss):
    loss = loss()
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    assert result.valid, "Test is flawed"
    params = loss.get_params(is_yield=False)
    param1 = list(params)[0]
    loss_values = result.fmin
    assert np.isfinite(loss_values)
    param1min = param1.value()
    param1.set_value(param1 *0.1)
    loss_values_nonmin = loss.value()
    assert np.isfinite(loss_values_nonmin), "Test is flawed, loss is not finite"
    assert loss_values_nonmin - loss_values > 0.1
    loss_values_params = loss.value(params={param1.name: param1min})
    assert pytest.approx(loss_values) == loss_values_params
    loss_values_result = loss.value(params=result)
    assert pytest.approx(loss_values) == loss_values_result
