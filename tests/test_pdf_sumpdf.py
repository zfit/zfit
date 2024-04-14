#  Copyright (c) 2024 zfit
import hist.axis
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pytest
import scipy.stats

import zfit
import zfit.z.numpy as znp
from zfit.util.exception import SpecificFunctionNotImplemented


def create_gaussians(yields):
    obs = zfit.Space("obs1", (-2, 5))

    gauss1 = zfit.pdf.Gauss(mu=2, sigma=4, obs=obs)
    gauss2 = zfit.pdf.Gauss(mu=3, sigma=5, obs=obs)
    gauss3 = zfit.pdf.Gauss(mu=1, sigma=6, obs=obs)
    gausses = gauss1, gauss2, gauss3

    if yields:
        if len(yields) == 2:
            yields = yields + [None]
        gausses = [
            gauss.create_extended(yiel) if yiel is not None else gauss
            for gauss, yiel in zip(gausses, yields)
        ]
    return gausses


def test_analytic_integral():
    obs = zfit.Space("obs1", (-2, 5))


@pytest.mark.parametrize("yields", [False, [4, 3, 5], [1, 2]])
def test_frac_behavior(yields):
    gauss1, gauss2, gauss3 = create_gaussians(yields)

    frac1 = zfit.Parameter("frac1", 0.4)
    frac2 = zfit.Parameter("frac2", 0.6)
    for fracs in [frac1, [frac1, frac2]]:
        sumpdf1 = zfit.pdf.SumPDF([gauss1, gauss2], fracs)
        assert sumpdf1.fracs[0] == frac1
        assert len(sumpdf1.fracs) == 2
        assert len(sumpdf1.params) == 2
        assert sumpdf1.params["frac_0"] == frac1
        assert sumpdf1.params["frac_1"] == sumpdf1.fracs[1]
        assert not sumpdf1.is_extended

        frac2_val = 1 - frac1.value()
        assert (
            pytest.approx(frac2_val) == sumpdf1.params["frac_1"].value()
        )
        if isinstance(fracs, list) and len(fracs) == 2:
            assert sumpdf1.params["frac_1"] == frac2

    frac1.set_value(0.3)
    frac3 = zfit.Parameter("frac3", 0.1)

    for fracs in [[frac1, frac2], [frac1, frac2, frac3]]:
        sumpdf2 = zfit.pdf.SumPDF([gauss1, gauss2, gauss3], fracs)
        assert sumpdf2.fracs[0] == frac1
        assert len(sumpdf2.fracs) == 3
        assert len(sumpdf2.params) == 3
        assert sumpdf2.params["frac_0"] == frac1
        assert sumpdf2.params["frac_1"] == frac2
        assert (
            pytest.approx(frac3.value())
            == sumpdf2.params["frac_2"].value()
        )
        assert not sumpdf1.is_extended

        if isinstance(fracs, list) and len(fracs) == 3:
            assert sumpdf2.params["frac_2"] == frac3


@pytest.mark.flaky(2)  # ks test
def test_sampling():
    class SimpleSampleSumPDF(zfit.pdf.SumPDF):
        @zfit.supports()
        def _sample(self, n, limits):
            raise SpecificFunctionNotImplemented  # fallback to the default sampling

    sample_size = 100_000
    tol = 0.1
    mu1, mu2 = 0, 10
    frac = 0.9
    true_mu = mu1 * frac + mu2 * (1 - frac)

    obs = zfit.Space("obs1", (mu1 - 5, mu2 + 5))
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=1)
    gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=1)

    sumpdf = zfit.pdf.SumPDF([gauss1, gauss2], frac)
    sumpdf_true = SimpleSampleSumPDF([gauss1, gauss2], frac)

    sample = sumpdf.sample(sample_size).value()[:, 0]
    sample_true = sumpdf_true.sample(sample_size).value()[:, 0]

    assert pytest.approx(
        np.mean(sample_true), abs=tol
    ) == true_mu  # if this is not True, it's a problem, the test is flawed
    assert pytest.approx(np.mean(sample), abs=tol) == true_mu
    assert pytest.approx(np.std(sample), abs=tol) == np.std(sample_true)

    plt.figure()
    plt.title("Sampling of SumPDF")
    axis = hist.axis.Regular(100, *obs.v1.limits)
    sample_hist = hist.Hist(axis).fill(sample)
    sample_true_hist = hist.Hist(axis).fill(sample_true)
    mplhep.histplot(sample_hist, density=True, label="sampled")
    mplhep.histplot(sample_true_hist, density=True, label="sampled true")
    plt.legend()
    pytest.zfit_savefig(folder="sampling")
    assert scipy.stats.mannwhitneyu(sample, sample_true).pvalue > 0.05


@pytest.mark.flaky(2)  # mc integration
def test_integrate():
    class SimpleSampleSumPDF(zfit.pdf.SumPDF):
        @zfit.supports()
        def _integrate(self, limits, norm, options):
            raise SpecificFunctionNotImplemented  # force fallback to the default sampling

        @zfit.supports()
        def _analytic_integrate(self, limits, norm):
            raise SpecificFunctionNotImplemented

        @zfit.supports()
        def _numeric_integrate(self, limits, norm=None, options=None):
            raise SpecificFunctionNotImplemented

    mu1, mu2 = 0, 1.7
    frac = 0.7

    lower = mu1 - 2.5
    upper = mu2 + 3
    obs = zfit.Space("obs1", (lower, upper))
    limits = zfit.Space("obs1", (mu1 - 0.3, mu2 + 0.1))
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=0.93)
    gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=1.2)

    sumpdf = zfit.pdf.SumPDF([gauss1, gauss2], frac)
    sumpdf_true = SimpleSampleSumPDF([gauss1, gauss2], frac)
    assert pytest.approx(1, abs=0.01) == (
        gauss1.integrate(
            limits,
            norm=limits,
        )
    )
    integral = sumpdf.integrate(
        limits=limits,
        norm=False,
    )
    integral_true = sumpdf_true.integrate(
        limits=limits,
        norm=False,
    )
    integral_manual_true = gauss1.integrate(
        limits,
    ) * frac + gauss2.integrate(
        limits,
    ) * (1 - frac)

    assert pytest.approx(integral_manual_true, rel=0.03) == integral_true
    assert pytest.approx(integral, rel=0.03) == integral_true
    assert integral_true < 0.85

    analytic_integral = sumpdf.analytic_integrate(limits=limits, norm=False)

    assert pytest.approx(analytic_integral, rel=0.03) == integral_true

    rnd_limits = [lower] + sorted(np.random.uniform(lower, upper, 16)) + [upper]
    integrals = [
        sumpdf.integrate((low, up), norm=False)
        for low, up in zip(rnd_limits[:-1], rnd_limits[1:])
    ]

    integral = np.sum(integrals)
    integral_full = sumpdf.integrate((lower, upper), norm=False)
    assert pytest.approx(integral_full) == integral
