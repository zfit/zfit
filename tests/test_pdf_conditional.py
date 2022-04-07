#  Copyright (c) 2022 zfit

import numpy as np
import pytest

import zfit


@pytest.mark.skipif(
    not zfit.run.get_graph_mode(),
    reason="Getting stuck for weird reason in eager. TODO.",
)
def test_conditional_pdf_simple():
    xobs = zfit.Space("x", (-2, 5))
    muobs = zfit.Space("y", (-1, 15))
    sigmaobs = zfit.Space("z", (0.2, 13))

    mu = zfit.Parameter("mu", 2, -5, 25)

    xmuobs = xobs * muobs
    obs = xobs * muobs * sigmaobs
    nsample = 5000
    uniform_sample = np.random.uniform(size=(nsample, 1), low=-1, high=4)
    normal_sample1 = np.random.normal(loc=2, scale=3.5, size=(nsample, 1))
    normal_sample2 = np.abs(np.random.normal(loc=5, scale=2, size=(nsample, 1))) + 0.9
    sigma = zfit.Parameter("sigma", 3, 0.1, 15)

    gauss = zfit.pdf.Gauss(obs=xobs, mu=mu, sigma=sigma)
    # gauss = zfit.pdf.Gauss(xmuobs=muobs, mu=mu, sigma=sigma)

    data2d = zfit.Data.from_numpy(
        array=np.stack([uniform_sample, normal_sample1], axis=-1), obs=xmuobs
    )
    data1d = zfit.Data.from_numpy(array=uniform_sample, obs=xobs)
    data3d = zfit.Data.from_numpy(
        array=np.stack([uniform_sample, normal_sample1, normal_sample2], axis=-1),
        obs=obs,
    )
    data1dmu = zfit.Data.from_numpy(array=uniform_sample, obs=muobs)

    cond_gauss2d = zfit.pdf.ConditionalPDFV1(
        pdf=gauss, cond={mu: muobs}, use_vectorized_map=False
    )
    prob2 = cond_gauss2d.pdf(data2d)
    assert prob2.shape[0] == data2d.nevents
    assert prob2.shape.rank == 1

    prob_check2 = np.ones_like(prob2)
    for i, (xval, muval) in enumerate(data2d.value()):
        mu.assign(muval)
        prob_check2[i] = gauss.pdf(xval)
    np.testing.assert_allclose(prob_check2, prob2, rtol=1e-5)

    cond_gauss1d = zfit.pdf.ConditionalPDFV1(pdf=gauss, cond={mu: xobs})
    prob1 = cond_gauss1d.pdf(data1d)
    assert prob1.shape[0] == data1d.nevents
    assert prob1.shape.rank == 1

    prob_check1 = np.ones_like(prob1)
    for i, xval in enumerate(data1d.value()[:, 0]):
        mu.assign(xval)
        prob_check1[i] = gauss.pdf(xval)
    np.testing.assert_allclose(prob_check1, prob1, rtol=1e-5)

    cond_gauss3d = zfit.pdf.ConditionalPDFV1(
        pdf=gauss, cond={mu: muobs, sigma: sigmaobs}
    )
    prob3 = cond_gauss3d.pdf(data3d)
    assert prob3.shape[0] == data3d.nevents
    assert prob3.shape.rank == 1

    prob_check3 = np.ones_like(prob3)
    for i, (xval, muval, sigmaval) in enumerate(data3d.value()):
        zfit.param.assign_values([mu, sigma], [muval, sigmaval])
        prob_check3[i] = gauss.pdf(xval)
    np.testing.assert_allclose(prob_check3, prob3, rtol=1e-5)

    nll = zfit.loss.UnbinnedNLL(model=cond_gauss2d, data=data2d)
    nll.value()

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    assert result.valid

    integrals2 = cond_gauss2d.integrate(limits=xobs, var=data2d)
    assert integrals2.shape[0] == data2d.nevents
    assert integrals2.shape.rank == 1
    np.testing.assert_allclose(integrals2, np.ones_like(integrals2), atol=1e-5)

    sample2 = cond_gauss2d.sample(n=data1dmu.nevents, limits=xobs, x=data1dmu)
    assert sample2.value().shape == (data1dmu.nevents, cond_gauss2d.n_obs)
