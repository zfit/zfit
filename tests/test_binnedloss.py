#  Copyright (c) 2023 zfit

import mplhep
import numpy as np
import pytest
from matplotlib import pyplot as plt

import zfit
import zfit.models.tobinned
import zfit.z.numpy as znp
from zfit import z
from zfit._data.binneddatav1 import BinnedData
from zfit._variables.axis import RegularBinning
from zfit.models.binned_functor import BinnedSumPDF
from zfit.models.template import BinnedTemplatePDFV1

mu_true = 1.2
sigma_true = 4.1
mu_true2 = 1.01
sigma_true2 = 3.5

mu_constr = [1.3, 0.02]  # mu, sigma
sigma_constr = [3.2, 0.03]

yield_true = 30000
test_values_np = np.random.normal(loc=mu_true, scale=sigma_true, size=(yield_true, 1))
test_values_np2 = np.random.normal(loc=mu_true2, scale=sigma_true2, size=yield_true)

low, high = -24.3, 28.6
obs1 = zfit.Space(
    "obs1",
    (
        np.min([test_values_np[:, 0], test_values_np2]) - 1.4,
        np.max([test_values_np[:, 0], test_values_np2]) + 2.4,
    ),
)


def create_params1(nameadd=""):
    mu1 = zfit.Parameter(
        "mu1" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 5.0, mu_true + 5.0
    )
    sigma1 = zfit.Parameter(
        "sigma1" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 4.0,
        sigma_true + 4.0,
    )
    return mu1, sigma1


def create_params2(nameadd=""):
    mu2 = zfit.Parameter(
        "mu25" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1.0, mu_true + 1.0
    )
    sigma2 = zfit.Parameter(
        "sigma25" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 2.0,
        sigma_true + 2.0,
    )
    return mu2, sigma2


def create_params3(nameadd=""):
    mu3 = zfit.Parameter(
        "mu35" + nameadd, z.to_real(mu_true) - 0.2, mu_true - 1.0, mu_true + 1.0
    )
    sigma3 = zfit.Parameter(
        "sigma35" + nameadd,
        z.to_real(sigma_true) - 0.3,
        sigma_true - 2.0,
        sigma_true + 2.0,
    )
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


@pytest.mark.parametrize(
    "Loss", [zfit.loss.ExtendedBinnedNLL, zfit.loss.ExtendedBinnedChi2]
)
def test_binned_extended_simple(Loss):
    # zfit.run.set_graph_mode(False)
    counts = np.random.uniform(high=1, size=(10, 20))  # generate counts
    counts2 = np.random.normal(loc=5, size=(10, 20))
    counts3 = np.linspace(0, 10, num=10)[:, None] * np.linspace(0, 5, num=20)[None, :]
    binning = [
        zfit.binned.VariableBinning(
            [-10, -5, 5, 7, 10, 15, 17, 21, 23.5, 27, 30], name="obs1"
        ),
        RegularBinning(20, 0, 10, name="obs2"),
    ]
    obs = zfit.Space(obs=["obs1", "obs2"], binning=binning)

    mc1 = BinnedData.from_tensor(
        space=obs, values=counts, variances=znp.ones_like(counts) * 1.3
    )
    mc2 = BinnedData.from_tensor(obs, counts2)
    mc3 = BinnedData.from_tensor(obs, counts3)
    sum_counts = counts + counts2 + counts3
    observed_data = BinnedData.from_tensor(
        space=obs, values=sum_counts, variances=(sum_counts + 0.5) ** 2
    )

    pdf = BinnedTemplatePDFV1(data=mc1)
    pdf2 = BinnedTemplatePDFV1(data=mc2)
    pdf3 = BinnedTemplatePDFV1(data=mc3)
    pdf_sum = BinnedSumPDF(pdfs=[pdf, pdf2, pdf3], obs=obs)

    nll = Loss(pdf_sum, data=observed_data)
    nll2 = Loss(pdf_sum, data=observed_data)
    nll.value(), nll.gradient()  # TODO: add some check?

    nllsum = nll + nll2  # check that sum works
    # TODO: should this actually work I think?
    # assert float(nllsum.value()) == pytest.approx(nll.value() + nll2.value(), rel=1e-3)


@pytest.mark.plots
@pytest.mark.flaky(2)
@pytest.mark.parametrize(
    "weights",
    [None, np.random.normal(loc=1.0, scale=0.2, size=test_values_np.shape[0])],
    ids=["no_weights", "weights_normal"],
)
@pytest.mark.parametrize(
    "Loss",
    [
        zfit.loss.BinnedNLL,
        zfit.loss.BinnedChi2,
        zfit.loss.ExtendedBinnedNLL,
        zfit.loss.ExtendedBinnedChi2,
    ],
)
@pytest.mark.parametrize(
    "simultaneous", [True, False], ids=["simultaneous", "sequential"]
)
def test_binned_loss(weights, Loss, simultaneous):
    obs = zfit.Space("obs1", limits=(-15, 25))
    gaussian1, mu1, sigma1 = create_gauss1(obs=obs)
    gaussian2, mu2, sigma2 = create_gauss2(obs=obs)
    test_values_np_shifted = test_values_np - 1.8  # shift them a bit
    test_values_np_shifted *= 1.2
    test_values = znp.array(test_values_np_shifted)
    test_values = zfit.Data.from_tensor(obs=obs, tensor=test_values, weights=weights)
    init_yield = test_values_np.shape[0] * 1.2
    scale = zfit.Parameter("yield", init_yield, 0, init_yield * 4, step_size=1)
    binning = zfit.binned.RegularBinning(92, obs.lower[0], obs.upper[0], name="obs1")
    obs_binned = obs.with_binning(binning)
    test_values_binned = test_values.to_binned(obs_binned)
    binned_gauss = zfit.pdf.BinnedFromUnbinnedPDF(gaussian1, obs_binned, extended=scale)
    binned_gauss_alt = gaussian2.to_binned(obs_binned, extended=scale)
    counts = binned_gauss.counts()
    counts_alt = binned_gauss_alt.counts()
    assert np.allclose(counts, counts_alt)
    binned_gauss_closure = binned_gauss.to_binned(obs_binned)
    assert np.allclose(counts, binned_gauss_closure.counts())
    if simultaneous:
        obs_binned2 = obs.with_binning(14)
        test_values_binned2 = test_values.to_binned(obs_binned2)
        binned_gauss2 = zfit.pdf.BinnedFromUnbinnedPDF(
            gaussian1, obs_binned2, extended=scale
        )
        loss = Loss(
            [binned_gauss, binned_gauss2],
            data=[test_values_binned, test_values_binned2],
        )

    else:
        loss = Loss(model=binned_gauss, data=test_values_binned)

    title = (
        f"Binned gaussian fit"
        f"{' (randomly weighted)' if weights is not None else ''} with "
        f"{loss.name}"
    )
    plt.figure()
    plt.title(title)
    mplhep.histplot(binned_gauss.to_hist(), label="PDF before fit")
    mplhep.histplot(test_values_binned.to_hist(), label="Data")

    # timing, uncomment to test
    # loss.value_gradient(params=loss.get_params())
    # loss.value()
    # loss.gradient()
    # import time, progressbar
    # start = time.time()
    # for _ in progressbar.progressbar(range(1000)):
    #     loss.value()
    #     loss.gradient()
    # print(f"Needed: {time.time() - start}")

    minimizer = zfit.minimize.Minuit(gradient=False)
    result = minimizer.minimize(loss=loss)

    params = result.params
    mplhep.histplot(binned_gauss.to_hist(), label="PDF after fit")
    plt.legend()
    pytest.zfit_savefig()

    result.hesse(name="hesse")
    result.errors(name="asymerr")
    str(result)  # check if no error
    rel_tol_errors = 0.1
    mu_error = 0.03 if not simultaneous else 0.021
    sigma_error = 0.0156 if simultaneous else 0.022
    params_list = [mu1, sigma1]
    errors = [mu_error, sigma_error]
    if loss.is_extended:
        params_list.append(scale)
        errors.append(122 if simultaneous else 170)
    for param, errorval in zip(params_list, errors):
        assert (
            pytest.approx(result.params[param]["hesse"]["error"], rel=rel_tol_errors)
            == errorval
        )
        assert (
            pytest.approx(result.params[param]["asymerr"]["lower"], rel=rel_tol_errors)
            == -errorval
        )
        assert (
            pytest.approx(result.params[param]["asymerr"]["upper"], rel=rel_tol_errors)
            == errorval
        )

    abs_tol_val = 0.15 if weights is None else 0.08  # more fluctuating with weights
    abs_tol_val *= 2 if isinstance(loss, zfit.loss.BinnedChi2) else 1

    assert params[mu1]["value"] == pytest.approx(
        np.mean(test_values_np_shifted), abs=abs_tol_val
    )
    assert params[sigma1]["value"] == pytest.approx(
        np.std(test_values_np_shifted), abs=abs_tol_val
    )
    if loss.is_extended:
        nexpected = test_values_np_shifted.shape[0]
        assert params[scale]["value"] == pytest.approx(
            nexpected, abs=3 * nexpected**0.5
        )
    constraints = zfit.constraint.GaussianConstraint(
        params=[mu2, sigma2],
        observation=[mu_constr[0], sigma_constr[0]],
        uncertainty=[mu_constr[1], sigma_constr[1]],
    )
    gaussian2 = zfit.models.tobinned.BinnedFromUnbinnedPDF(
        gaussian2, obs_binned, extended=scale
    )
    loss = Loss(model=gaussian2, data=test_values_binned, constraints=constraints)

    minimizer = zfit.minimize.Minuit(gradient=False)
    result = minimizer.minimize(loss=loss, params=[mu2, sigma2])
    params = result.params
    if weights is None:
        assert params[mu2]["value"] > np.mean(test_values_np_shifted)
        assert params[sigma2]["value"] < np.std(test_values_np_shifted)


@pytest.mark.parametrize("Loss", [zfit.loss.BinnedChi2, zfit.loss.ExtendedBinnedChi2])
@pytest.mark.parametrize(
    "empty", [None, "ignore", False], ids=["empty", "ignore", "False"]
)
@pytest.mark.parametrize(
    "errors",
    [None, "expected", "data"],
    ids=["error_default", "error_expected", "error_data"],
)
def test_binned_chi2_loss(Loss, empty, errors):  # TODO: add test with zeros in bins
    obs = zfit.Space("obs1", limits=(-1, 2))
    gaussian1, mu1, sigma1 = create_gauss1(obs=obs)
    test_values_np_shifted = test_values_np - 1.8  # shift them a bit
    test_values_np_shifted *= 1.2
    test_values = znp.array(test_values_np_shifted)
    test_values = zfit.Data.from_tensor(obs=obs, tensor=test_values)
    init_yield = test_values_np.shape[0] * 1.2
    scale = zfit.Parameter("yield", init_yield, 0, init_yield * 4, step_size=1)
    binning = zfit.binned.RegularBinning(32, obs.lower[0], obs.upper[0], name="obs1")
    obs_binned = obs.with_binning(binning)
    test_values_binned = test_values.to_binned(obs_binned)
    binned_gauss = zfit.models.tobinned.BinnedFromUnbinnedPDF(
        gaussian1, obs_binned, extended=scale
    )

    loss = Loss(
        model=binned_gauss,
        data=test_values_binned,
        options={"empty": empty, "errors": errors},
    )
    loss.value_gradient(loss.get_params())


@pytest.mark.parametrize(
    "weights",
    [None, np.random.normal(loc=1.0, scale=0.2, size=test_values_np.shape[0])],
    ids=["weights_none", "weights_random"],
)
@pytest.mark.parametrize(
    "Loss",
    [
        zfit.loss.BinnedNLL,
        zfit.loss.BinnedChi2,
        zfit.loss.ExtendedBinnedNLL,
        zfit.loss.ExtendedBinnedChi2,
    ],
)
def test_binned_loss_hist(weights, Loss):
    obs = zfit.Space("obs1", limits=(-15, 25))
    gaussian1, mu1, sigma1 = create_gauss1(obs=obs)
    test_values_np_shifted = test_values_np - 1.8  # shift them a bit
    test_values_np_shifted *= 1.2
    test_values = znp.array(test_values_np_shifted)
    test_values = zfit.Data.from_tensor(obs=obs, tensor=test_values, weights=weights)
    init_yield = test_values_np.shape[0] * 1.2
    scale = zfit.Parameter("yield", init_yield, 0, init_yield * 4, step_size=1)
    binning = zfit.binned.RegularBinning(32, obs.lower[0], obs.upper[0], name="obs1")
    obs_binned = obs.with_binning(binning)
    test_values_binned = test_values.to_binned(obs_binned)
    h = test_values_binned.to_hist()
    binned_gauss = zfit.models.tobinned.BinnedFromUnbinnedPDF(
        gaussian1, obs_binned, extended=scale
    )

    loss = Loss(model=binned_gauss, data=h)
    loss2 = Loss(model=binned_gauss, data=test_values_binned)

    assert pytest.approx(float(loss.value())) == float(loss2.value())

    nllsum = loss + loss2  # check that sum works
    nllsum += loss2  # check that sum works
    # TODO: this should actually work I think
    # assert nllsum.value() == pytest.approx(loss.value() + loss2.value(), rel=1e-3)
