#  Copyright (c) 2022 zfit
import pytest

import zfit
from zfit.util.exception import NormRangeUnderdefinedError

limits1 = (-4, 3)
limits2 = (-2, 5)
limits3 = (-1, 7)
obs1 = "obs1"
obs2 = "obs2"

space1 = zfit.Space(obs=obs1, limits=limits1)
space2 = zfit.Space(obs=obs1, limits=limits2)
space3 = zfit.Space(obs=obs1, limits=limits3)
space4 = zfit.Space(obs=obs2, limits=limits2)

space5 = space1.combine(space4)


def test_norm_range():
    gauss1 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss2 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss3 = zfit.pdf.Gauss(1.0, 4.0, obs=space2)

    sum1 = zfit.pdf.SumPDF(pdfs=[gauss1, gauss2], fracs=0.4, obs=space1)
    assert sum1.obs == (obs1,)
    assert sum1.norm == space1

    with pytest.raises(NormRangeUnderdefinedError):
        _ = zfit.pdf.SumPDF(pdfs=[gauss1, gauss3], fracs=0.34)
    sum2 = zfit.pdf.SumPDF(pdfs=[gauss1, gauss3], fracs=0.34, obs=space3)

    sum2.set_norm_range(space2)
    with sum2.set_norm_range(space3):
        assert sum2.norm == space3
    assert sum2.norm == space2


def test_combine_range():
    gauss1 = zfit.pdf.Gauss(1.0, 4.0, obs=space1)
    gauss4 = zfit.pdf.Gauss(1.0, 4.0, obs=space4)
    gauss5 = zfit.pdf.Gauss(1.0, 4.0, obs=space4)

    product = zfit.pdf.ProductPDF(pdfs=[gauss1, gauss4])
    assert product.obs == (obs1, obs2)
    assert product.norm == space5

    product = zfit.pdf.ProductPDF(pdfs=[gauss1, gauss4, gauss5])
    assert product.obs == (obs1, obs2)
    assert product.norm == space5
