#  Copyright (c) 2020 zfit

import numpy as np

import zfit


def test_conditional_pdf_simple():
    zfit.run.set_graph_mode(True)
    xobs = zfit.Space('x', (-2, 5))
    muobs = zfit.Space('y', (-1, 15))

    mu = zfit.Parameter('mu', 2, -5, 25)
    sigma = zfit.Parameter('sigma', 5, 1, 10)

    gauss = zfit.pdf.Gauss(obs=xobs, mu=mu, sigma=sigma)

    obs = xobs * muobs
    data = zfit.Data.from_numpy(array=np.random.uniform(size=(50000, 2)), obs=obs)

    cond_gauss = zfit.pdf.ConditionalPDFV1(pdf=gauss, cond={mu: muobs})
    prob = cond_gauss.pdf(data)
