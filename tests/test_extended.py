#  Copyright (c) 2024 zfit

import numpy as np
import pytest

import zfit
from zfit.core.sample import extended_sampling, extract_extended_pdfs

obs1 = zfit.Space("obs1", limits=(-3, 4))


@pytest.mark.flaky(reruns=3)  # poissonian sampling
def test_extract_extended_pdfs():
    gauss1 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss2 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss3 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss4 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss5 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss6 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)

    yield1 = zfit.Parameter("yield123" + str(np.random.random()), 200.0)

    # sum1 = 0.3 * gauss1 + gauss2
    gauss3_ext = gauss3.create_extended(45)
    gauss4_ext = gauss4.create_extended(100)
    sum2_ext_daughters = gauss3_ext + gauss4_ext
    sum3 = zfit.pdf.SumPDF((gauss5, gauss6), 0.4)
    sum3_ext = sum3.create_extended(yield1)

    sum_all = zfit.pdf.SumPDF(pdfs=[sum2_ext_daughters, sum3_ext], norm=(-5, 5))


    extracted_pdfs = extract_extended_pdfs(pdfs=sum_all)
    assert frozenset(extracted_pdfs) == {gauss3_ext, gauss4_ext, sum3_ext}

    limits = zfit.Space(obs=obs1, limits=(-4, 5))
    limits = limits.with_autofill_axes()
    extended_sample = extended_sampling(pdfs=sum_all, limits=limits)
    assert pytest.approx(
        expected=(45 + 100 + 200), rel=0.1
    ) == np.shape(extended_sample)[0]
    samples_from_pdf = sum_all.sample(n="extended", limits=limits).value()
    assert pytest.approx(
        expected=(45 + 100 + 200), rel=0.1
    ) == np.shape(samples_from_pdf)[0]


def test_set_yield():
    gauss6 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)

    yield1 = zfit.Parameter("yield123" + str(np.random.random()), 200.0)
    assert not gauss6.is_extended
    gauss6.set_yield(yield1)
    assert gauss6.is_extended


def test_create_extended_keeps_space_and_norm():
    # the observable space must not be replaced by the normalization range
    obs = zfit.Space("obs_ext_norm", limits=(-10, 10))
    norm = zfit.Space("obs_ext_norm", limits=(0, 10))
    x = np.array([-2.0, 1.0, 3.0])

    gauss = zfit.pdf.Gauss(mu=0.0, sigma=1.0, obs=obs, norm=norm)
    gauss_ext = gauss.create_extended(500)
    assert gauss_ext.is_extended
    assert gauss_ext.space == obs
    assert gauss_ext.norm == norm
    np.testing.assert_allclose(gauss_ext.pdf(x), gauss.pdf(x))
    np.testing.assert_allclose(gauss_ext.ext_pdf(x), 500 * gauss.pdf(x))

    gauss_copy = gauss.copy()
    assert gauss_copy.space == obs
    assert gauss_copy.norm == norm

    # without a separate norm, space and norm stay the same
    gauss_plain = zfit.pdf.Gauss(mu=0.0, sigma=1.0, obs=obs)
    gauss_plain_ext = gauss_plain.create_extended(500)
    assert gauss_plain_ext.space == obs
    assert gauss_plain_ext.norm == obs

    # also for a sum of PDFs
    gauss2 = zfit.pdf.Gauss(mu=1.0, sigma=2.0, obs=obs, norm=norm)
    sum_pdf = zfit.pdf.SumPDF([gauss, gauss2], fracs=0.3, norm=norm)
    sum_ext = sum_pdf.create_extended(500)
    assert sum_ext.space == sum_pdf.space
    assert sum_ext.norm == norm
