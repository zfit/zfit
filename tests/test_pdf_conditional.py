#  Copyright (c) 2020 zfit

import numpy as np

import zfit


def test_conditional_pdf_simple():
    # zfit.run.set_graph_mode(False)
    xobs = zfit.Space('x', (-2, 5))
    muobs = zfit.Space('y', (-1, 15))
    sigmaobs = zfit.Space('z', (-1.5, 13))

    mu = zfit.Parameter('mu', 2, -5, 25)

    xmuobs = xobs * muobs
    obs = xobs * muobs * sigmaobs
    nsample = 5000
    uniform_sample = np.random.uniform(size=(nsample, 1))
    normal_sample1 = np.random.normal(loc=2, scale=0.5, size=(nsample, 1))
    normal_sample2 = np.random.normal(loc=5, scale=2, size=(nsample, 1))
    sigma = zfit.Parameter('sigma', 5, 0.1, 10)

    gauss = zfit.pdf.Gauss(obs=xobs, mu=mu, sigma=sigma)
    # gauss = zfit.pdf.Gauss(xmuobs=muobs, mu=mu, sigma=sigma)

    data2d = zfit.Data.from_numpy(array=np.stack([normal_sample1, uniform_sample], axis=-1),
                                  obs=xmuobs)
    data1d = zfit.Data.from_numpy(array=normal_sample1, obs=xobs)
    data3d = zfit.Data.from_numpy(array=np.stack([uniform_sample, normal_sample1, normal_sample2], axis=-1),
                                  obs=obs)

    cond_gauss2d = zfit.pdf.ConditionalPDFV1(pdf=gauss, cond={mu: muobs})
    prob2 = cond_gauss2d.pdf(data2d)
    assert prob2.shape[0] == data2d.nevents
    assert prob2.shape.rank == 1

    cond_gauss1d = zfit.pdf.ConditionalPDFV1(pdf=gauss, cond={mu: xobs})
    prob1 = cond_gauss1d.pdf(data1d)
    assert prob1.shape[0] == data1d.nevents
    assert prob1.shape.rank == 1

    cond_gauss3d = zfit.pdf.ConditionalPDFV1(pdf=gauss, cond={mu: muobs, sigma: sigmaobs})
    prob3 = cond_gauss3d.pdf(data3d)
    assert prob3.shape[0] == data3d.nevents
    assert prob3.shape.rank == 1

    nll = zfit.loss.UnbinnedNLL(model=cond_gauss2d, data=data2d)
    nll.value()

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    assert result.valid

    integrals2 = cond_gauss2d.integrate(limits=xobs, x=data2d)
    assert integrals2.shape[0] == data2d.nevents
    assert integrals2.shape.rank == 1
