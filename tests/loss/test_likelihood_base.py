#  Copyright (c) 2024 zfit
import numpy as np
import pytest

import zfit

def test_binned_nll_simple_v2():
    import zfit._loss.likelihood
    zfit.run.set_graph_mode(False)

    obs_binned = zfit.Space("obs1", lower=0, upper=1000, binning=2)
    databinned = zfit.Data.from_numpy(obs=obs_binned, array=np.random.normal(500, 100, 10000))
    model = zfit.pdf.Gauss(mu=zfit.Parameter("mu", 500, 400, 600),
                            sigma=zfit.Parameter("sigma", 100, 1, 200),
                            obs=obs_binned)
    nll = zfit._loss.likelihood.BinnedNLL(expected=model, observed=databinned)
    nll_legacy = zfit.loss.BinnedNLL(model=model, data=databinned)
    assert pytest.approx(nll_legacy.value(), rel=1e-5) == nll()
    # minimizer = zfit.minimize.Minuit(verbosity=7)
    # result = minimizer.minimize(nll)
    # print(result)
