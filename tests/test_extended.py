#  Copyright (c) 2019 zfit

from zfit.core.testing import setup_function, teardown_function, tester


import pytest
import numpy as np
import tensorflow as tf

import zfit
from zfit.core.sample import extract_extended_pdfs, extended_sampling
from zfit.core.testing import setup_function, teardown_function, tester

obs1 = 'obs1'


@pytest.mark.flaky(reruns=3)  # poissonian sampling
def test_extract_extended_pdfs():
    gauss1 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss2 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss3 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss4 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss5 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)
    gauss6 = zfit.pdf.Gauss(obs=obs1, mu=1.3, sigma=5.4)

    yield1 = zfit.Parameter('yield123' + str(np.random.random()), 200.)

    # sum1 = 0.3 * gauss1 + gauss2
    gauss3_ext = 45. * gauss3
    gauss4_ext = 100. * gauss4
    sum2_ext_daughters = gauss3_ext + gauss4_ext
    sum3 = 0.4 * gauss5 + gauss6
    sum3_ext = sum3.create_extended(yield1)

    sum_all = zfit.pdf.SumPDF(pdfs=[sum2_ext_daughters, sum3_ext])
    sum_all.set_norm_range((-5, 5))

    extracted_pdfs = extract_extended_pdfs(pdfs=sum_all)
    assert frozenset(extracted_pdfs) == {gauss3_ext, gauss4_ext, sum3_ext}

    limits = zfit.Space(obs=obs1, limits=(-4, 5))
    limits = limits.with_autofill_axes()
    extended_sample = extended_sampling(pdfs=sum_all, limits=limits)
    extended_sample_np = zfit.run(extended_sample)
    assert np.shape(extended_sample_np)[0] == pytest.approx(expected=(45 + 100 + 200), rel=0.1)
    samples_from_pdf = sum_all.sample(n='extended', limits=limits)
    samples_from_pdf_np = zfit.run(samples_from_pdf)
    assert np.shape(samples_from_pdf_np)[0] == pytest.approx(expected=(45 + 100 + 200), rel=0.1)
