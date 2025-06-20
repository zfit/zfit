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

def test_chi2_simple_v2():
    import zfit._loss.likelihood
    zfit.run.set_graph_mode(False)
    obs_binned = zfit.Space("obs1", lower=0, upper=1000, binning=15)
    databinned = zfit.Data.from_numpy(obs=obs_binned, array=np.random.normal(500, 100, 10000))
    model = zfit.pdf.Gauss(mu=zfit.Parameter("mu", 500, 400, 600),
                            sigma=zfit.Parameter("sigma", 100, 1, 200),
                            obs=obs_binned)
    chi2 = zfit._loss.likelihood.Chi2Half(expected=model, observed=databinned)
    chi2_legacy = zfit.loss.BinnedChi2(model=model, data=databinned)
    value = chi2()
    legacy_value = chi2_legacy.value(full=True)
    print(value, legacy_value)
    assert pytest.approx(legacy_value, rel=1e-5) == value * 2
