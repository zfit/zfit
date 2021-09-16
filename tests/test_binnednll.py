#  Copyright (c) 2021 zfit

import mplhep
import numpy as np
import pytest
from matplotlib import pyplot as plt

import zfit
import zfit.z.numpy as znp
from zfit import z
from zfit._data.binneddatav1 import BinnedDataV1
from zfit._variables.axis import Regular
from zfit.models.binned_functor import BinnedSumPDFV1
from zfit.models.template import BinnedTemplatePDFV1

mu_true = 1.2
sigma_true = 4.1
mu_true2 = 1.01
sigma_true2 = 3.5

mu_constr = [1.3, 0.2]  # mu, sigma
sigma_constr = [3.6, 0.2]

yield_true = 3000
test_values_np = np.random.normal(loc=mu_true, scale=sigma_true, size=(yield_true, 1))
test_values_np2 = np.random.normal(loc=mu_true2, scale=sigma_true2, size=yield_true)

low, high = -24.3, 28.6
obs1 = zfit.Space('obs1', (np.min([test_values_np[:, 0], test_values_np2]) - 1.4,
                           np.max([test_values_np[:, 0], test_values_np2]) + 2.4))


def create_params1(nameadd=""):
    mu1 = zfit.Parameter("mu1" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
    sigma1 = zfit.Parameter("sigma1" + nameadd, z.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
    return mu1, sigma1


def create_params2(nameadd=""):
    mu2 = zfit.Parameter("mu25" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
    sigma2 = zfit.Parameter("sigma25" + nameadd, z.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
    return mu2, sigma2


def create_params3(nameadd=""):
    mu3 = zfit.Parameter("mu35" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1., mu_true + 1.)
    sigma3 = zfit.Parameter("sigma35" + nameadd, z.to_real(sigma_true) - 0.3, sigma_true - 2., sigma_true + 2.)
    yield3 = zfit.Parameter("yield35" + nameadd, yield_true + 300, 0, 10000000)
    return mu3, sigma3, yield3


def create_gauss1(obs=None):
    if obs is None:
        obs = obs1
    mu, sigma = create_params1()
    return zfit.pdf.Gauss(mu, sigma, obs=obs, name="gaussian1"), mu, sigma


def create_gauss2(obs=None):
    if obs is None:
        obs = obs1
    mu, sigma = create_params2()
    return zfit.pdf.Gauss(mu, sigma, obs=obs, name="gaussian2"), mu, sigma


def test_binned_nll_simple():
    # zfit.run.set_graph_mode(False)
    counts = np.random.uniform(high=1, size=(10, 20))  # generate counts
    counts2 = np.random.normal(loc=5, size=(10, 20))
    counts3 = np.linspace(0, 10, num=10)[:, None] * np.linspace(0, 5, num=20)[None, :]
    binning = [
        zfit.binned.Variable([-10, -5, 5, 7, 10, 15, 17, 21, 23.5, 27, 30], name='obs1'),
        Regular(20, 0, 10, name='obs2'),
    ]
    obs = zfit.Space(obs=['obs1', 'obs2'], binning=binning)

    mc1 = BinnedDataV1.from_tensor(space=obs, values=counts, variances=znp.ones_like(counts) * 1.3)
    mc2 = BinnedDataV1.from_tensor(obs, counts2)
    mc3 = BinnedDataV1.from_tensor(obs, counts3)
    sum_counts = counts + counts2 + counts3
    observed_data = BinnedDataV1.from_tensor(space=obs, values=sum_counts, variances=(sum_counts + 0.5) ** 2)

    pdf = BinnedTemplatePDFV1(data=mc1)
    pdf2 = BinnedTemplatePDFV1(data=mc2)
    pdf3 = BinnedTemplatePDFV1(data=mc3)
    # assert len(pdf.ext_pdf(None)) > 0
    pdf_sum = BinnedSumPDFV1(pdfs=[pdf, pdf2, pdf3], obs=obs)

    nll = zfit.loss.ExtendedBinnedNLL(pdf_sum, data=observed_data)
    print(nll.value())


@pytest.mark.plots
@pytest.mark.flaky(3)
@pytest.mark.parametrize('weights', [None, np.random.normal(loc=1., scale=0.2, size=test_values_np.shape[0])])
# @pytest.mark.parametrize('weights', [None])
def test_binned_nll(weights):
    zfit.run.set_autograd_mode(False)
    obs = zfit.Space("obs1", limits=(-15, 25))
    gaussian1, mu1, sigma1 = create_gauss1(obs=obs)
    gaussian2, mu2, sigma2 = create_gauss2(obs=obs)
    test_values_np_shifted = test_values_np - 0.8  # shift them a bit
    test_values = znp.array(test_values_np_shifted)
    test_values = zfit.Data.from_tensor(obs=obs, tensor=test_values, weights=weights)

    binning = zfit.binned.Regular(22, obs.lower[0], obs.upper[0], name="obs1")
    obs_binned = obs.with_binning(binning)
    test_values_binned = test_values.to_binned(obs_binned)
    binned_gauss = zfit.pdf.BinnedFromUnbinnedPDF(gaussian1, obs_binned)

    title = f"Binned gaussian fit{' with random weights' if weights is not None else ''}"
    plt.figure()
    plt.title(title)
    mplhep.histplot(binned_gauss.to_hist(), label="PDF before fit")
    mplhep.histplot(test_values_binned.to_hist() / float(test_values_binned.nevents),
                    label="Data")
    nll_object = zfit.loss.BinnedNLL(model=binned_gauss, data=test_values_binned)
    # nll_object.value_gradient(params=nll_object.get_params())
    # start = time.time()
    # for _ in progressbar.progressbar(range(100)):
    #     nll_object.value()
    # print(f"Needed: {time.time() - start}")
    minimizer = zfit.minimize.Minuit()
    status = minimizer.minimize(loss=nll_object, params=[mu1, sigma1])
    # status.hesse()
    # status.errors()
    params = status.params
    # plt.figure()
    mplhep.histplot(binned_gauss.to_hist(), label="PDF after fit")
    plt.legend()
    pytest.zfit_savefig()
    # plt.show()
    # mplhep.histplot(test_values_binned.to_hist() /  float(test_values_binned.nevents))
    rel_error = 0.035 if weights is None else 0.15  # more fluctuating with weights

    assert params[mu1]['value'] == pytest.approx(np.mean(test_values_np_shifted), abs=rel_error)
    assert params[sigma1]['value'] == pytest.approx(np.std(test_values_np_shifted), abs=rel_error)

    constraints = zfit.constraint.nll_gaussian(params=[mu2, sigma2],
                                               observation=[mu_constr[0], sigma_constr[0]],
                                               uncertainty=[mu_constr[1], sigma_constr[1]])
    gaussian2 = zfit.pdf.BinnedFromUnbinnedPDF(gaussian2, obs_binned)
    nll_object = zfit.loss.BinnedNLL(model=gaussian2, data=test_values_binned,
                                     constraints=constraints)

    minimizer = zfit.minimize.Minuit()
    status = minimizer.minimize(loss=nll_object, params=[mu2, sigma2])
    params = status.params
    if weights is None:
        assert params[mu2]['value'] > np.mean(test_values_np_shifted)
        assert params[sigma2]['value'] < np.std(test_values_np_shifted)
