import pytest

import zfit
from zfit.util.exception import LimitsOverdefinedError

limits1 = (-4, 3)
limits2 = (-2, 5)
limits3 = (-1, 7)
obs1 = 'obs1'
space1 = zfit.Space(obs=obs1, limits=limits1)
space2 = zfit.Space(obs=obs1, limits=limits2)
space3 = zfit.Space(obs=obs1, limits=limits3)


def test_norm_range():
    gauss1 = zfit.pdf.Gauss(1., 4., obs=space1)
    gauss2 = zfit.pdf.Gauss(1., 4., obs=space1)
    gauss3 = zfit.pdf.Gauss(1., 4., obs=space2)

    sum1 = zfit.pdf.SumPDF(pdfs=[gauss1, gauss2], fracs=0.4)
    assert sum1.obs == (obs1,)
    assert sum1.norm_range == space1

    sum2 = zfit.pdf.SumPDF(pdfs=[gauss1, gauss3], fracs=0.34)
    with pytest.raises(LimitsOverdefinedError):
        sum2.norm_range

    sum2.set_norm_range(space2)
    with sum2.set_norm_range(space3):
        assert sum2.norm_range == space3
    assert sum2.norm_range == space2
