#  Copyright (c) 2020 zfit
import pytest

import zfit
# noinspection PyUnresolvedReferences
from zfit.core.testing import setup_function, teardown_function, tester


def create_gaussians(yields):
    obs = zfit.Space('obs1', (-2, 5))

    gauss1 = zfit.pdf.Gauss(mu=2, sigma=4, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu=3, sigma=5, obs=obs)
    gauss3 = zfit.pdf.Gauss(mu=1, sigma=6, obs=obs)
    gausses = gauss1, gauss2, gauss3

    if yields:
        if len(yields) == 2:
            yields = yields + [None]
        gausses = [gauss.create_extended(yiel) if yiel is not None else gauss
                   for gauss, yiel in zip(gausses, yields)]
    return gausses


def test_analytic_integral():
    obs = zfit.Space('obs1', (-2, 5))


@pytest.mark.parametrize('yields', [False, [4, 3, 5], [1, 2]])
def test_frac_behavior(yields):
    gauss1, gauss2, gauss3 = create_gaussians(yields)

    frac1 = zfit.Parameter('frac1', 0.4)
    frac2 = zfit.Parameter('frac2', 0.6)
    for fracs in [frac1, [frac1, frac2]]:
        sumpdf1 = zfit.pdf.SumPDF([gauss1, gauss2], fracs)
        assert sumpdf1.fracs[0] == frac1
        assert len(sumpdf1.fracs) == 2
        assert len(sumpdf1.params) == 2
        assert sumpdf1.params['frac_0'] == frac1
        assert sumpdf1.params['frac_1'] == sumpdf1.fracs[1]
        assert not sumpdf1.is_extended

        frac2_val = 1 - frac1.value()
        assert pytest.approx(frac2_val.numpy(), sumpdf1.params['frac_1'].value().numpy())
        if isinstance(fracs, list) and len(fracs) == 2:
            assert sumpdf1.params['frac_1'] == frac2

    frac1.set_value(0.3)
    frac3 = zfit.Parameter('frac3', 0.1)

    for fracs in [[frac1, frac2], [frac1, frac2, frac3]]:
        sumpdf2 = zfit.pdf.SumPDF([gauss1, gauss2, gauss3], fracs)
        assert sumpdf2.fracs[0] == frac1
        assert len(sumpdf2.fracs) == 3
        assert len(sumpdf2.params) == 3
        assert sumpdf2.params['frac_0'] == frac1
        assert sumpdf2.params['frac_1'] == frac2
        assert pytest.approx(frac3.value().numpy(), sumpdf2.params['frac_2'].value().numpy())
        assert not sumpdf1.is_extended

        if isinstance(fracs, list) and len(fracs) == 3:
            assert sumpdf2.params['frac_2'] == frac3
