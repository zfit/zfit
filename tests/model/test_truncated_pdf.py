#  Copyright (c) 2024 zfit

import pytest

import numpy as np

import zfit


@pytest.fixture
def pdf():
    from zfit.pdf import Gauss

    mu = zfit.Parameter("mu", 0)
    sigma = zfit.Parameter("sigma", 1)
    return Gauss(mu=mu, sigma=sigma, obs=zfit.Space("obs1", (-10, 10)))


@pytest.fixture
def limits() -> list[zfit.Space]:
    return [zfit.Space("obs1", (-1, 1)), zfit.Space("obs1", (2, 3))]


def test_truncated_pdf_initialization(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    assert truncated_pdf.pdfs[0] == pdf
    assert truncated_pdf._limits == limits


def test_truncated_pdf_unnormalized_pdf(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    x = np.array([-0.5, 0.5, 2.5])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob >= 0)


def test_truncated_pdf_normalization(pdf, limits):
    truncated_pdf = zfit.pdf.TruncatedPDF(pdf, limits)
    norm = truncated_pdf.normalization(limits[0])
    assert np.all(norm >= 0)


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
    samples = truncated_pdf.sample(n, limits[0]).numpy()
    assert np.all(samples >= -1)
    assert np.all(samples <= 1)

    samples = truncated_pdf.sample(n, limits[1]).numpy()
    assert np.all(samples >= 2)
    assert np.all(samples <= 3)

    samples = truncated_pdf.sample(n, space)[pdf.obs[0]].numpy()
    all_inside = (samples >= -1) * (samples <= 1) + (samples >= 2) * (samples <= 3)
    np.testing.assert_array_equal(all_inside, np.ones(n, dtype=bool))

    space2 = zfit.Space("obs1", (-5, 3))
    samples = truncated_pdf.sample(n, space2)[pdf.obs[0]].numpy()
    all_inside = (samples >= -1) * (samples <= 1) + (samples >= 2) * (samples <= 3)
    np.testing.assert_array_equal(all_inside, np.ones(n, dtype=bool))
