#  Copyright (c) 2024 zfit

import pytest
from zfit.models.piecewise import TruncatedPDF
from zfit import Parameter, Space
import numpy as np


@pytest.fixture
def pdf():
    from zfit.pdf import Gauss

    mu = Parameter("mu", 0)
    sigma = Parameter("sigma", 1)
    return Gauss(mu=mu, sigma=sigma, obs=Space("obs1", (-10, 10)))


@pytest.fixture
def limits() -> list[Space]:
    return [Space("obs1", (-1, 1)), Space("obs1", (2, 3))]


def test_truncated_pdf_initialization(pdf, limits):
    truncated_pdf = TruncatedPDF(pdf, limits)
    assert truncated_pdf.pdfs[0] == pdf
    assert truncated_pdf._limits == limits


def test_truncated_pdf_unnormalized_pdf(pdf, limits):
    truncated_pdf = TruncatedPDF(pdf, limits)
    x = np.array([-0.5, 0.5, 2.5])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob >= 0)


def test_truncated_pdf_normalization(pdf, limits):
    truncated_pdf = TruncatedPDF(pdf, limits)
    norm = truncated_pdf.normalization(limits[0])
    assert np.all(norm >= 0)


def test_truncated_pdf_outside_limits(pdf, limits):
    truncated_pdf = TruncatedPDF(pdf, limits)
    x = np.array([-2, 1.5, 4])
    prob = truncated_pdf.pdf(x)
    assert np.all(prob == 0)
