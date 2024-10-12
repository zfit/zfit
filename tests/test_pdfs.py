#  Copyright (c) 2024 zfit

from __future__ import annotations

from typing import TYPE_CHECKING

from zfit import z

if TYPE_CHECKING:
    import zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
import zfit.z.numpy as znp
from zfit.core.data import Data
from zfit.core.interfaces import ZfitPDF, ZfitBinnedPDF
from zfit.core.parameter import Parameter
from zfit.core.space import Space
from zfit.models.dist_tfp import Gauss
from zfit.models.functor import ProductPDF, SumPDF

low, high = -0.64, 5.9

mu1_true = 1.0
mu2_true = 2.0
mu3_true = 0.6
sigma1_true = 1.4
sigma2_true = 2.3
sigma3_true = 1.8

fracs = [0.3, 0.15]


def makeobs1():
    return zfit.Space("obs1", (-3, 6))


def makeobs2():
    return zfit.Space("obs2", (-2, 5))


def makeobs3():
    return zfit.Space("obs3", (-3.5, 4))


def makeobs4():
    return zfit.Space("obs4", (-2.6, 4.1))


def true_gaussian_sum(x):
    def norm(sigma):
        return np.sqrt(2 * np.pi) * sigma

    sum_gauss = (
            fracs[0]
            * np.exp(-((x - mu1_true) ** 2) / (2 * sigma1_true ** 2))
            / norm(sigma1_true)
    )
    sum_gauss += (
            fracs[1]
            * np.exp(-((x - mu2_true) ** 2) / (2 * sigma2_true ** 2))
            / norm(sigma2_true)
    )
    sum_gauss += (
            (1.0 - sum(fracs))
            * np.exp(-((x - mu3_true) ** 2) / (2 * sigma3_true ** 2))
            / norm(sigma3_true)
    )
    return sum_gauss


def create_params(name_add=""):
    mu1 = Parameter("mu1" + name_add, mu1_true)
    mu2 = Parameter("mu2" + name_add, mu2_true)
    mu3 = Parameter("mu3" + name_add, mu3_true)
    sigma1 = Parameter("sigma1" + name_add, sigma1_true)
    sigma2 = Parameter("sigma2" + name_add, sigma2_true)
    sigma3 = Parameter("sigma3" + name_add, sigma3_true)
    return mu1, mu2, mu3, sigma1, sigma2, sigma3


def create_gaussians(norms=None) -> list[ZfitPDF]:
    # Gauss for sum, same axes
    if norms is None:
        norms = [None] * 3
    obs1 = makeobs1()
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params()
    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1asum", norm=norms[0])
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss2asum", norm=norms[1])
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss3asum", norm=norms[2])
    return [gauss1, gauss2, gauss3]


def sum_gaussians():
    obs1 = makeobs1()
    gauss_dists = create_gaussians()
    sum_gauss = SumPDF(pdfs=gauss_dists, fracs=fracs, obs=obs1)
    return sum_gauss


def product_gaussians():
    obs1 = makeobs1()
    gauss_dists = create_gaussians()
    prod_gauss = ProductPDF(pdfs=gauss_dists, obs=obs1)
    return prod_gauss


def product_gauss_3d(name="", binned=None):
    # Gauss for product, independent
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params("3d" + name)
    obs1 = makeobs1()
    obs2 = makeobs2()
    obs3 = makeobs3()
    gauss13 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1a")
    gauss23 = Gauss(mu=mu2, sigma=sigma2, obs=obs2, name="gauss2a")
    gauss33 = Gauss(mu=mu3, sigma=sigma3, obs=obs3, name="gauss3a")
    gauss_dists3 = [gauss13, gauss23, gauss33]
    space = None
    if binned:
        space = obs1 * obs2 * obs3
        space = space.with_binning([70, 113, 89])
    prod_gauss_3d = ProductPDF(pdfs=gauss_dists3, obs=space)
    if binned:
        assert isinstance(prod_gauss_3d, ZfitBinnedPDF)
    return prod_gauss_3d

    # prod_gauss_3d.update_integration_options(draws_per_dim=33)


def test_product_separation():
    obs1, obs2, obs3, obs4 = makeobs1(), makeobs2(), makeobs3(), makeobs4()
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params()
    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss1asum")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs2, name="gauss2asum")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss3asum")
    gauss4 = Gauss(mu=mu3, sigma=sigma3, obs=obs3, name="gauss3asum")
    gauss5 = Gauss(mu=mu3, sigma=sigma3, obs=obs4, name="gauss3asum")

    prod12 = ProductPDF(pdfs=[gauss1, gauss2, gauss3])
    assert not prod12._prod_is_same_obs_pdf
    prod13 = ProductPDF(pdfs=[gauss3, gauss4])
    assert not prod13._prod_is_same_obs_pdf
    prod123 = ProductPDF([prod12, prod13])
    assert prod123._prod_is_same_obs_pdf
    npoints = 2000
    data3 = zfit.Data.from_numpy(array=np.linspace(0, 1, npoints), obs=obs3)
    data2 = zfit.Data.from_numpy(array=np.linspace(0, 1, npoints), obs=obs2)
    data1 = zfit.Data.from_numpy(array=np.linspace(0, 1, npoints), obs=obs1)
    data4 = zfit.Data.from_numpy(array=np.linspace(0, 1, npoints), obs=obs4)
    integral13 = prod13.partial_integrate(x=data3, limits=obs1, norm=False)
    assert integral13.shape[0] == npoints
    trueint3 = gauss4.pdf(data3, norm=False) * gauss3.integrate(obs1, norm=False)
    np.testing.assert_allclose(integral13, trueint3)
    assert (
            prod12.partial_integrate(
                x=data2,
                limits=obs1,
            ).shape[0]
            == npoints
    )
    assert (
            prod123.partial_integrate(
                x=data3,
                limits=obs1 * obs2,
            ).shape[0]
            == npoints
    )
    assert (
            prod123.partial_integrate(
                x=data2,
                limits=obs1 * obs3,
            ).shape[0]
            == npoints
    )

    prod1234 = ProductPDF(pdfs=[gauss1, gauss2, gauss4, gauss5])
    integ = prod1234.partial_integrate(data1, limits=obs2 * obs3 * obs4, norm=False)
    assert integ.shape[0] == npoints

    obs13 = obs1 * obs3
    analytic_int = znp.asarray(prod13.analytic_integrate(limits=obs13, norm=False))
    numeric_int = znp.asarray(prod13.numeric_integrate(limits=obs13, norm=False))
    assert pytest.approx(analytic_int, rel=1e-3) == numeric_int


def product_gauss_4d():
    mu1, mu2, mu3, sigma1, sigma2, sigma3 = create_params("4d")
    obs1, obs3 = makeobs1(), makeobs3()
    obs4 = zfit.Space("obs4", (-4.5, 4.7))
    gauss12 = Gauss(mu=mu1, sigma=sigma1, obs=obs4, name="gauss12a")
    gauss22 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss22a")
    gauss32 = Gauss(mu=mu3, sigma=sigma3, obs=obs3, name="gauss32a")
    gauss_dists2 = [gauss12, gauss22, gauss32]
    obs = zfit.Space(["obs1", "obs2", "obs3", "obs4"])

    prod_3d = product_gauss_3d("4d")
    prod_gauss_4d = ProductPDF(pdfs=gauss_dists2 + [prod_3d], obs=obs)
    return prod_gauss_4d


@pytest.mark.parametrize("binned", [False, True], ids=["unbinned", "binned"])
def test_prod_gauss_nd(binned):
    obs1, obs2, obs3 = makeobs1(), makeobs2(), makeobs3()
    test_values = np.random.random(size=(10, 3))
    test_values_data = Data.from_tensor(obs=obs1 * obs2 * obs3, tensor=test_values)
    product_pdf = product_gauss_3d(binned=binned)
    if not binned:
        assert len(product_pdf._prod_disjoint_obs_pdfs) == 3
        assert product_pdf._prod_is_same_obs_pdf is False
    assert product_pdf.n_obs == 3
    probs = product_pdf.pdf(x=test_values_data)
    gaussians = create_gaussians()
    # raise RuntimeError("Cleanup test!")
    norms = [obs1, obs2, obs3]


    true_probs = np.prod(
        [gauss.pdf(test_values[:, i], norm=norm.v1.limits) for i, (gauss, norm) in enumerate(zip(gaussians, norms))], axis=0
    )

    rtol = 1e-5
    if binned:
        rtol = 0.1  # we bin it, so we don't expect a high precision
    np.testing.assert_allclose(true_probs, probs, rtol=rtol)


@pytest.mark.flaky(reruns=3)
def test_prod_gauss_nd_mixed():
    obs1, obs2, obs3 = makeobs1(), makeobs2(), makeobs3()
    norm = (-5, 4)
    low, high = norm
    test_values = np.random.uniform(low=low, high=high, size=(1000, 4))

    obs4d = ["obs1", "obs2", "obs3", "obs4"]
    test_values_data = Data.from_tensor(obs=obs4d, tensor=test_values)
    limits_4d = Space(limits=(((-5,) * 4,), ((4,) * 4,)), obs=obs4d)
    prod_gauss_4d = product_gauss_4d()
    probs = prod_gauss_4d.pdf(x=test_values_data, norm=limits_4d)
    gausses = create_gaussians(norms=[norm] * 3)

    gauss1, gauss2, gauss3 = gausses
    prod_gauss_3d = product_gauss_3d()

    def probs_4d(values):
        true_prob = [gauss1.pdf(values[:, 3])]
        true_prob += [gauss2.pdf(values[:, 0])]
        true_prob += [gauss3.pdf(values[:, 2])]
        true_prob += [
            prod_gauss_3d.pdf(
                values[:, 0:3],
                norm=Space(
                    limits=(((-5,) * 3,), ((4,) * 3,)), obs=["obs1", "obs2", "obs3"]
                ),
            )
        ]
        return tf.math.reduce_prod(true_prob, axis=0)

    true_unnormalized_probs = probs_4d(values=test_values)

    normalization_probs = limits_4d.volume * probs_4d(
        z.random.uniform(minval=low, maxval=high, shape=(40 ** 4, 4))
    )
    true_probs = true_unnormalized_probs / tf.reduce_mean(
        input_tensor=normalization_probs
    )
    assert pytest.approx(
        1.0, rel=0.33
    ) == np.average(probs * limits_4d._legacy_area())  # low n mc
    np.testing.assert_allclose(true_probs, probs, rtol=2e-2)


def test_func_sum():
    obs1 = makeobs1()
    sum_gauss = sum_gaussians()
    test_values = np.random.uniform(low=-3, high=4, size=10)
    sum_gauss_as_func = sum_gauss.as_func(norm=(-10, 10))
    vals = sum_gauss_as_func.func(x=test_values)
    vals = znp.asarray(vals)
    # test_sum = sum([g.func(test_values) for g in gauss_dists])
    np.testing.assert_allclose(
        vals, true_gaussian_sum(test_values), rtol=3e-2
    )  # MC integral


def test_normalization_sum_gauss():
    normalization_testing(sum_gaussians())


def test_normalization_sum_gauss_extended():
    test_yield = 109.0
    sum_gauss_extended = sum_gaussians().create_extended(test_yield)
    normalization_testing(sum_gauss_extended)


def test_normalization_prod_gauss():
    normalization_testing(product_gaussians())


def test_exp():
    lambda_true = 0.031
    lambda_ = zfit.Parameter("lambda1", lambda_true)
    exp1 = zfit.pdf.Exponential(lam=lambda_, obs=zfit.Space("obs1", (-11, 11)))
    sample = exp1.sample(n=1000, limits=(-10, 10)).value()
    assert not any(np.isnan(sample))

    exp2 = zfit.pdf.Exponential(lam=lambda_, obs=zfit.Space("obs1", (5250, 5750)))
    probs2 = exp2.pdf(x=np.linspace(5300, 5700, num=1100))
    assert not any(np.isnan(probs2))
    normalization_testing(exp2, limits=(5400, 5800))

    intlim = [5400, 5500]
    integral2 = znp.asarray(
        exp2.integrate(
            intlim,
        )
    )
    numintegral2 = exp2.numeric_integrate(intlim)
    assert pytest.approx(numintegral2, rel=0.03) == integral2


def normalization_testing(pdf, limits=None):
    import zfit.z.numpy as znp

    limits = (low, high) if limits is None else limits
    space = Space(obs=makeobs1(), limits=limits)

    samples = znp.random.uniform(
        low=space.v1.lower, high=space.v1.upper, size=(100_000, pdf.n_obs)
    )

    samples = zfit.Data.from_tensor(obs=space, tensor=samples)
    probs = pdf.pdf(samples, norm=space)
    result = znp.asarray(np.average(probs) * space.volume)
    assert pytest.approx(result, rel=0.03) == 1


def test_extended_gauss():
    obs1 = makeobs1()
    mu1 = Parameter("mu11", 1.0)
    mu2 = Parameter("mu21", 12.0)
    mu3 = Parameter("mu31", 3.0)
    sigma1 = Parameter("sigma11", 1.0)
    sigma2 = Parameter("sigma21", 12.0)
    sigma3 = Parameter("sigma31", 33.0)
    yield1 = Parameter("yield11", 150.0)
    yield2 = Parameter("yield21", 550.0)
    yield3 = Parameter("yield31", 2500.0)

    gauss1 = Gauss(mu=mu1, sigma=sigma1, obs=obs1, name="gauss11")
    gauss2 = Gauss(mu=mu2, sigma=sigma2, obs=obs1, name="gauss21")
    gauss3 = Gauss(mu=mu3, sigma=sigma3, obs=obs1, name="gauss31")
    gauss1 = gauss1.create_extended(yield1)
    gauss2 = gauss2.create_extended(yield2)
    gauss3 = gauss3.create_extended(yield3)

    gauss_dists = [gauss1, gauss2, gauss3]

    sum_gauss = SumPDF(pdfs=gauss_dists)
    integral_true = (
            sum_gauss.integrate(
                (-1, 5),
            )
            * sum_gauss.get_yield()
    )

    assert pytest.approx(
        sum_gauss.ext_integrate(
            (-1, 5),
        )) == integral_true
    normalization_testing(pdf=sum_gauss, limits=obs1)
