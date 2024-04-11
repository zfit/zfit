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
