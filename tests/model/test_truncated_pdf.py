#  Copyright (c) 2024 zfit

import pytest

import numpy as np

import zfit
import zfit.z.numpy as znp


@pytest.fixture
def pdf():
    from zfit.pdf import Gauss

    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    return Gauss(mu=mu, sigma=sigma, obs=zfit.Space("obs1", (-10, 10)))


@pytest.fixture
def extpdf():
    from zfit.pdf import Gauss

    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    return Gauss(mu=mu, sigma=sigma, obs=zfit.Space("obs1", (-10, 10)), extended=zfit.Parameter("extended", 1004))


@pytest.fixture
def limits() -> list[zfit.Space]:
    return [zfit.Space("obs1", (-1, 1)), zfit.Space("obs1", (2, 3))]


def test_truncated_pdf_initialization(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits=limits)
    assert truncated_pdf.pdfs[0] == pdf
    assert truncated_pdf._limits == tuple(limits)


def test_truncated_pdf_unnormalized_pdf(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    x = np.array([-0.5, 0.5, 2.5])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob >= 0)
    # test all outside limits
    x = np.array([-2, 1.5, 4])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob == 0)


def test_truncated_pdf_normalization(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    norm = truncated_pdf.normalization(limits[0])
    assert np.all(norm >= 0)


@pytest.mark.parametrize("wrapper", [True, False], ids=["wrapper", "no_wrapper"])
def test_truncated_pdf_normalization_ext(extpdf, pdf, limits, wrapper):
    if wrapper:
        truncated_pdf = zfit.pdf.TruncatedPDF(extpdf, limits=limits)
    else:
        truncated_pdf = extpdf.to_truncated(limits=limits)
    x = znp.linspace(*limits[0].v1.limits, num=100)
    prob = truncated_pdf.ext_pdf(x)
    prob_true = extpdf.ext_pdf(x)
    np.testing.assert_allclose(prob, prob_true)
    assert truncated_pdf.extended
    assert pytest.approx(truncated_pdf.ext_integrate(limits[0])) == extpdf.ext_integrate(limits[0])
    norm = truncated_pdf.normalization(limits[0])
    assert np.all(norm >= 0)

@pytest.mark.parametrize("limits", [None, True], ids=["limits_none", "equivalent_limits"])
def test_pdf_to_truncated(extpdf, limits):
    if limits:
        lim = [extpdf.space.v1.lower, -2, 4, 4.5, extpdf.space.v1.upper]

        limits = [zfit.Space("obs1", lower=lim[i], upper=lim[i + 1]) for i in range(0, len(lim) - 1)]
    truncated_pdf = extpdf.to_truncated(limits=limits)
    space = extpdf.space
    assert truncated_pdf.integrate(space) == extpdf.integrate(space)
    x = np.array([-0.5, 0.5, 2.5])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob >= 0)
    # test all outside limits
    x = np.array([space.v1.lower -5, space.v1.upper +3])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob == 0)
    sample = truncated_pdf.sample(n=1000000)
    sample_true = extpdf.sample(n=1000000)
    assert pytest.approx(np.mean(sample.value()), abs=0.01) == np.mean(sample_true.value())
    assert pytest.approx(np.std(sample.value()), abs=0.01) == np.std(sample_true.value())



def test_truncated_pdf_outside_limits(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    x = np.array([-2, 1.5, 4])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob == 0)


def test_truncated_pdf_integrate(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    integral = truncated_pdf.integrate(limits[0])
    assert np.all(integral >= 0)


def test_truncated_pdf_sample(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    space = pdf.space
    n = 10_000
    samples = truncated_pdf.sample(n, limits[0]).value()
    assert np.all(samples >= -1)
    assert np.all(samples <= 1)

    samples = truncated_pdf.sample(n, limits[1]).value()
    assert np.all(samples >= 2)
    assert np.all(samples <= 3)

    samples = truncated_pdf.sample(n, space)[pdf.obs[0]]
    samples = np.array(samples)
    all_inside = (samples >= -1) * (samples <= 1) + (samples >= 2) * (samples <= 3)
    np.testing.assert_array_equal(all_inside, np.ones(n, dtype=bool))

    space2 = zfit.Space("obs1", (-5, 3))
    samples = np.array(truncated_pdf.sample(n, space2)[pdf.obs[0]])
    all_inside = (samples >= -1) * (samples <= 1) + (samples >= 2) * (samples <= 3)
    np.testing.assert_array_equal(all_inside, np.ones(n, dtype=bool))

def test_dynamic_truncated_yield():
    import numpy as np

    import zfit

    obs = zfit.Space("x", -10, 10)

    mean = 3
    obs1 = zfit.Space("x", -10, mean)
    obs2 = zfit.Space("x", mean, 10)


    # parameters
    mu_shared = zfit.Parameter("mu_shared", 2.0, -4, 6)  # mu is a shared parameter
    sigma1 = zfit.Parameter("sigma_one", 1.0, 0.1, 10)
    globyield = zfit.Parameter("yield1", 1100, 500, 2000)


    # model building, pdf creation
    gauss1 = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma1, obs=obs, extended=globyield)
    gauss2 = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma1, obs=obs, extended=globyield)

    # data
    normal_np = np.random.normal(loc=2.0, scale=3.0, size=1000)
    data = zfit.Data.from_numpy(obs=obs, array=normal_np)
    data1 = zfit.Data.from_numpy(obs=obs1, array=normal_np)  # data for the first model, can also be just a numpy array
    # the data objects just makes sure that the data is within the limits

    mean = 5
    gauss1_trunc = gauss1.to_truncated(limits=(-10, mean))
    gauss2_trunc = gauss1.to_truncated(limits=(mean, 10))

    # data
    data2 = zfit.Data.from_numpy(obs=obs2, array=normal_np)


    nll1 = zfit.loss.ExtendedUnbinnedNLL(model=gauss1_trunc, data=data1)
    nll2 = zfit.loss.ExtendedUnbinnedNLL(model=gauss2_trunc, data=data2)
    nll_simultaneous2 = nll1 + nll2

    minimizer = zfit.minimize.Minuit(tol=1e-4, gradient='zfit')
    result = minimizer.minimize(nll_simultaneous2)
    result.hesse()

    ntrue = data1.nevents + data2.nevents
    assert pytest.approx(result.params[globyield]['value'], abs=0.3) == ntrue

    nll_norm = zfit.loss.ExtendedUnbinnedNLL(model=gauss1, data=data)
    result = minimizer.minimize(nll_norm)
    result.hesse()
    assert pytest.approx(result.params[globyield]['value'], abs=0.3) == ntrue
