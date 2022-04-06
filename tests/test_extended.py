#  Copyright (c) 2022 zfit

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

    sum_all = zfit.pdf.SumPDF(pdfs=[sum2_ext_daughters, sum3_ext])
    sum_all.set_norm_range((-5, 5))

    extracted_pdfs = extract_extended_pdfs(pdfs=sum_all)
    assert frozenset(extracted_pdfs) == {gauss3_ext, gauss4_ext, sum3_ext}

    limits = zfit.Space(obs=obs1, limits=(-4, 5))
    limits = limits.with_autofill_axes()
    extended_sample = extended_sampling(pdfs=sum_all, limits=limits)
    extended_sample_np = extended_sample.numpy()
    assert np.shape(extended_sample_np)[0] == pytest.approx(
        expected=(45 + 100 + 200), rel=0.1
    )
    samples_from_pdf = sum_all.sample(n="extended", limits=limits)
    samples_from_pdf_np = samples_from_pdf.numpy()
    assert np.shape(samples_from_pdf_np)[0] == pytest.approx(
        expected=(45 + 100 + 200), rel=0.1
    )


def test_set_yield():
    gauss6 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)

    yield1 = zfit.Parameter("yield123" + str(np.random.random()), 200.0)
    assert not gauss6.is_extended
    gauss6.set_yield(yield1)
    assert gauss6.is_extended
