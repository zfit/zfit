#  Copyright (c) 2023 zfit
import time

import hist
import mplhep
import numpy as np
import pytest
from matplotlib import pyplot as plt

import zfit.pdf
import zfit.z.numpy as znp
from zfit.models.interpolation import SplinePDF
from zfit.models.tobinned import BinnedFromUnbinnedPDF


@pytest.mark.plots
def test_spline_from_binned_from_unbinned():
    n = 1004
    gauss, gauss_binned, obs, obs_binned = create_gauss_binned(n)

    x = znp.linspace(-5, 10, n // 5)

    # values = gauss_binned.rel_counts(obs_binned)

    sample = gauss_binned.sample(n, limits=obs_binned)

    title = "Comparison of binned gaussian and sample"
    plt.figure()
    plt.title(title)
    mplhep.histplot(sample.to_hist(), label="sampled binned")
    plt.plot(
        obs_binned.binning.centers[0],
        gauss_binned.counts(obs_binned),
        label="counts binned",
    )
    plt.legend()
    pytest.zfit_savefig()

    spline_gauss = SplinePDF(gauss_binned, obs=obs)
    y = spline_gauss.ext_pdf(x)
    y_true = gauss.ext_pdf(x)
    plt.figure()
    plt.title("Comparison of unbinned gauss to binned to interpolated")
    plt.plot(
        obs_binned.binning.centers[0],
        gauss_binned.ext_pdf(obs_binned),
        "x",
        label="binned",
    )
    plt.plot(x, y_true, label="original")
    plt.plot(x, y, ".", label="interpolated")
    plt.legend()
    pytest.zfit_savefig()

    np.testing.assert_allclose(y, y_true, atol=50)


def test_spline2D_from_binned_from_unbinned():
    n = 1204
    gauss, gauss_binned, obs, obs_binned = create_gauss2d_binned(n, 13)

    x = znp.random.uniform([-5, 50], [10, 600], size=(1000, 2))
    data = zfit.Data.from_tensor(obs, x)

    spline_gauss = SplinePDF(gauss_binned, obs=obs)
    y = spline_gauss.ext_pdf(data)
    y_true = gauss.ext_pdf(data)

    np.testing.assert_allclose(y, y_true, atol=50)


@pytest.mark.plots
def test_unbinned_from_binned_from_unbinned():
    n = 1004
    gauss, gauss_binned, obs, obs_binned = create_gauss_binned(n)

    x = znp.linspace(-5, 10, n // 5)

    # values = gauss_binned.rel_counts(obs_binned)

    sample = gauss_binned.sample(n, limits=obs_binned)

    title = "Comparison of binned gaussian and sample"
    plt.figure()
    plt.title(title)
    mplhep.histplot(sample.to_hist(), label="sampled binned")
    plt.plot(
        obs_binned.binning.centers[0],
        gauss_binned.counts(obs_binned),
        label="counts binned",
    )
    plt.legend()
    pytest.zfit_savefig()

    unbinned = zfit.pdf.UnbinnedFromBinnedPDF(gauss_binned, obs=obs)
    unbinned2 = gauss_binned.to_unbinned()
    y = unbinned.ext_pdf(x)
    y2 = unbinned2.ext_pdf(x)
    assert np.allclose(y, y2)
    y3 = unbinned.to_unbinned().ext_pdf(x)
    assert np.allclose(y, y3)
    y_true = gauss.ext_pdf(x)
    plt.figure()
    plt.title("Comparison of unbinned gauss to binned to unbinned again")
    plt.plot(
        obs_binned.binning.centers[0],
        gauss_binned.ext_pdf(obs_binned),
        "x",
        label="binned",
    )
    plt.plot(x, y_true, label="original")
    plt.plot(x, y, ".", label="unbinned")
    plt.legend()
    pytest.zfit_savefig()
    np.testing.assert_allclose(y, y_true, atol=50)

    nsample = 500000
    sample_binned = unbinned.sample(nsample).to_binned(obs_binned)
    sample_binned_hist = sample_binned.to_hist()
    sample_gauss = gauss.sample(nsample).to_binned(obs_binned)
    sample_gauss_hist = sample_gauss.to_hist()

    title = "Comparison of unbinned gaussian and unbinned from binned sampled"
    plt.figure()
    plt.title(title)
    mplhep.histplot(sample_binned_hist, label="unbinned from binned")
    mplhep.histplot(sample_gauss_hist, label="original")
    plt.legend()
    pytest.zfit_savefig()

    diff = (sample_binned_hist.values() - sample_gauss_hist.values()) / (
        sample_gauss_hist.variances() + 1
    ) ** 0.5
    np.testing.assert_allclose(diff, 0, atol=7)  # 7 sigma away


def test_2D_unbinned_from_binned_from_unbinned():
    n = 1204
    gauss, gauss_binned, obs, obs_binned = create_gauss2d_binned(n, 13)

    x = znp.random.uniform([-5, 50], [10, 600], size=(1000, 2))
    data = zfit.Data.from_tensor(obs, x)

    unbinned = zfit.pdf.UnbinnedFromBinnedPDF(gauss_binned, obs=obs)
    y = unbinned.ext_pdf(x)
    y_true = gauss.ext_pdf(data)

    np.testing.assert_allclose(y, y_true, atol=50)

    nsample = 500000
    unbinned_sample = unbinned.sample(nsample)
    sample_binned = unbinned_sample.to_binned(obs_binned)
    sample_binned_hist = sample_binned.to_hist()
    sample_gauss = gauss.sample(nsample).to_binned(obs_binned)
    sample_gauss_hist = sample_gauss.to_hist()

    diff = (sample_binned_hist.values() - sample_gauss_hist.values()) / (
        sample_gauss_hist.variances() + 1
    ) ** 0.5
    np.testing.assert_allclose(diff, 0, atol=7)  # 7 sigma away


@pytest.mark.plots
def test_unbinned_data():
    n = 751
    gauss, gauss_binned, obs, obs_binned = create_gauss_binned(n, 70)
    x = znp.linspace(-5, 10, 200)
    centers = obs_binned.binning.centers[0]
    y_binned = gauss_binned.pdf(x)
    y_true = gauss.pdf(x)
    max_error = np.max(y_true) / 10
    np.testing.assert_allclose(y_true, y_binned, atol=max_error)

    ycenter_binned = gauss_binned.pdf(centers)
    ycenter_true = gauss.pdf(centers)
    np.testing.assert_allclose(ycenter_binned, ycenter_true, atol=max_error / 10)

    x_outside = znp.array([-7.0, 3.0, 12])
    y_outside = gauss_binned.pdf(x_outside)
    assert y_outside[0] == 0
    assert y_outside[1] > 0
    assert y_outside[2] == 0

    plt.figure()
    plt.title("Binned Gauss evaluated on unbinned edges")
    plt.plot(centers, ycenter_true, label="unbinned pdf")
    plt.plot(centers, ycenter_binned, "--", label="binned pdf")
    plt.legend()
    pytest.zfit_savefig()
    # plt.show()

    plt.figure()
    plt.title("Binned Gauss evaluated on unbinned data")
    plt.plot(x, y_true, label="unbinned pdf")
    plt.plot(x, y_binned, "--", label="binned pdf")
    plt.legend()
    pytest.zfit_savefig()


def test_unbinned_data2D():
    n = 751
    gauss, gauss_binned, obs, obs_binned = create_gauss2d_binned(n, 50)

    data = znp.random.uniform([-5, 50], [10, 600], size=(1000, 2))
    y_binned = gauss_binned.pdf(data)
    y_true = gauss.pdf(data)
    max_error = np.max(y_true) / 10
    np.testing.assert_allclose(y_true, y_binned, atol=max_error)

    centers = obs_binned.binning.centers
    X, Y = znp.meshgrid(*centers, indexing="ij")
    centers = znp.stack([znp.reshape(t, (-1,)) for t in (X, Y)], axis=-1)
    ycenter_binned = gauss_binned.pdf(centers)
    ycenter_true = gauss.pdf(centers)
    np.testing.assert_allclose(ycenter_binned, ycenter_true, atol=max_error / 10)

    # for the extended case
    y_binned_ext = gauss_binned.ext_pdf(data)
    y_true_ext = gauss.ext_pdf(data)
    max_error_ext = np.max(y_true_ext) / 10
    np.testing.assert_allclose(y_true_ext, y_binned_ext, atol=max_error_ext)

    ycenter_binned_ext = gauss_binned.ext_pdf(centers)
    ycenter_true_ext = gauss.ext_pdf(centers)
    np.testing.assert_allclose(
        ycenter_binned_ext, ycenter_true_ext, atol=max_error_ext / 10
    )

    x_outside = znp.array([[-7.0, 55], [3.0, 13], [2, 150], [12, 30], [14, 1000]])
    y_outside = gauss_binned.pdf(x_outside)
    assert y_outside[0] == 0
    assert y_outside[1] == 0
    assert y_outside[2] > 0
    assert y_outside[3] == 0
    assert y_outside[4] == 0

    y_outside_ext = gauss_binned.ext_pdf(x_outside)
    assert y_outside_ext[0] == 0
    assert y_outside_ext[1] == 0
    assert y_outside_ext[2] > 0
    assert y_outside_ext[3] == 0
    assert y_outside_ext[4] == 0


def create_gauss_binned(n, nbins=130):
    mu = zfit.Parameter("mu", 1, 0, 19)
    sigma = zfit.Parameter("sigma", 1, 0, 19)
    obs = zfit.Space("x", (-5, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
    gauss.set_yield(n)
    axis = zfit.binned.RegularBinning(nbins, -5, 10, name="x")
    obs_binned = zfit.Space("x", binning=[axis])
    gauss_binned = BinnedFromUnbinnedPDF(pdf=gauss, space=obs_binned, extended=n)
    return gauss, gauss_binned, obs, obs_binned


def create_gauss2d_binned(n, nbins=130):
    mu = zfit.Parameter("mu", 1, 0, 19)
    sigma = zfit.Parameter("sigma", 1, 0, 19)
    obsx = zfit.Space("x", (-5, 10))
    obsy = zfit.Space("y", (50, 600))
    obs2d = obsx * obsy
    gaussx = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obsx)
    gaussy = zfit.pdf.Gauss(mu=250, sigma=200, obs=obsy)
    prod = zfit.pdf.ProductPDF([gaussx, gaussy])
    prod.set_yield(n)
    if isinstance(nbins, int):
        nbins = [nbins, nbins]
    axisx = zfit.binned.RegularBinning(nbins[0], -5, 10, name="x")
    axisy = zfit.binned.RegularBinning(nbins[1], 50, 600, name="y")
    obs_binned = zfit.Space(["x", "y"], binning=[axisx, axisy])
    gauss_binned = BinnedFromUnbinnedPDF(pdf=prod, space=obs_binned, extended=n)
    return prod, gauss_binned, obs2d, obs_binned


# TODO(Jonas): add test for binned pdf with unbinned data
# def test_binned_with_unbinned_data():
#     n = 100000
#
#     mu = zfit.Parameter("mu", 1, 0, 19)
#     sigma = zfit.Parameter("sigma", 1, 0, 19)
#     obs = zfit.Space("x", (-5, 10))
#     gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
#     gauss.set_yield(n)
#     axis = zfit.binned.RegularBinning(130, -5, 10, name="x")
#     obs_binned = zfit.Space("x", binning=[axis])
#     gauss_binned = BinnedFromUnbinnedPDF(pdf=gauss, space=obs_binned, extended=n)
#
#     data = znp.random.uniform(-5, 10, size=(1000,))
#     y_binned = gauss_binned.pdf(data)
#     # check shape
#     assert y_binned.shape[0] == data.shape[0]


def test_binned_from_unbinned_2D():
    n = 100000

    mu = zfit.Parameter("mu", 1, 0, 19)
    sigma = zfit.Parameter("sigma", 6, 0, 120)
    obsx = zfit.Space("x", (-5, 10))
    obsy = zfit.Space("y", (-50, 100))
    gaussx = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obsx)
    muy = mu + 3
    sigmay = sigma * 20
    gaussy = zfit.pdf.Gauss(mu=muy, sigma=sigmay, obs=obsy)
    gauss2D = zfit.pdf.ProductPDF([gaussx, gaussy])

    axisx = zfit.binned.VariableBinning(
        np.concatenate([np.linspace(-5, 5, 43), np.linspace(5, 10, 30)[1:]], axis=0),
        name="x",
    )
    axisxhist = hist.axis.Variable(
        np.concatenate([np.linspace(-5, 5, 43), np.linspace(5, 10, 30)[1:]], axis=0),
        name="x",
    )
    axisy = zfit.binned.RegularBinning(15, -50, 100, name="y")
    axisyhist = hist.axis.Regular(15, -50, 100, name="y")
    obs_binnedx = zfit.Space(["x"], binning=axisx)
    obs_binnedy = zfit.Space("y", binning=axisy)
    obs_binned = obs_binnedx * obs_binnedy

    gauss_binned = BinnedFromUnbinnedPDF(pdf=gauss2D, space=obs_binned, extended=n)
    values = gauss_binned.rel_counts(obs_binned)  # TODO: good test?
    start = time.time()
    ntrial = 10
    for _ in range(ntrial):
        values = gauss_binned.rel_counts(obs_binned)
    # print(f"Time taken {(time.time() - start) / ntrial}")
    hist2d = hist.Hist(axisxhist, axisyhist)
    nruns = 5
    npoints = 5_000_000
    for _ in range(nruns):
        normal2d = np.random.normal(
            [float(mu), float(muy)], [float(sigma), float(sigmay)], size=(npoints, 2)
        )
        hist2d.fill(*normal2d.T, threads=4)

    diff = np.abs(values * hist2d.sum() - hist2d.counts()) - 6.5 * np.sqrt(
        hist2d.counts()
    )  # 5 sigma for 1000 bins
    np.testing.assert_array_less(diff, 0)

    sample = gauss_binned.sample(n, limits=obs_binned)
    hist_sampled = sample.to_hist()
    hist_pdf = gauss_binned.to_hist()
    max_error = hist_sampled.values() * 6**2  # 6 sigma away
    np.testing.assert_array_less(
        (hist_sampled.values() - hist_pdf.values()) ** 2, max_error
    )
    plt.figure()
    plt.title("Gauss 2D binned sampled.")
    mplhep.hist2dplot(hist_sampled)
    pytest.zfit_savefig()
    plt.figure()
    plt.title("Gauss 2D binned plot, irregular (x<4.5 larger bins than x>4.5) binning.")
    mplhep.hist2dplot(hist_pdf)
    pytest.zfit_savefig()


@pytest.mark.parametrize(
    "ndim",
    [
        1,
        2,
        3,
    ],
    ids=lambda x: f"{x}D",
)
def test_binned_sampler(ndim):
    nbins = 134
    if ndim == 1:
        dims = (nbins,)
        gauss = create_gauss_binned(n=100000, nbins=dims[0])
        gauss = gauss[1]
    elif ndim == 2:
        dims = (nbins, 57)
        gauss = create_gauss2d_binned(n=100000, nbins=dims)
        obs2d = gauss[3]
        gauss = gauss[1]
    elif ndim == 3:
        dims = (nbins, 5, 3)
        obs = (
            zfit.Space("x", (-5, 10))
            * zfit.Space("y", (1, 5))
            * zfit.Space("z", (3, 6))
        )
        gauss = np.random.normal(
            loc=[0.5, 1.5, 3.6], scale=[1.2, 2.1, 0.4], size=(100000, ndim)
        )

        data = zfit.data.Data.from_numpy(obs=obs, array=gauss).to_binned(dims)
        gauss = zfit.pdf.HistogramPDF(data=data, extended=True)
    else:
        raise ValueError("ndim must be 1 or 2")
    nsampled = 100000
    sample = gauss.sample(n=nsampled)
    sampler = gauss.create_sampler(n=nsampled)

    assert sample.values().shape == dims
    assert sampler.values().shape == dims
    assert np.sum(sample.values()) == pytest.approx(nsampled)
    assert np.sum(sampler.values()) == pytest.approx(nsampled)

    sampler.resample(n=nsampled * 2)
    assert sampler.values().shape == dims
    assert np.sum(sampler.values()) == pytest.approx(nsampled * 2)

    if ndim == 2:
        sampler_swapped = sampler.with_obs(
            obs=obs2d.with_obs([obs2d.obs[1], obs2d.obs[0]])
        )
        values_swapped = sampler_swapped.values()
        assert values_swapped.shape == (dims[1], dims[0])
        assert np.sum(values_swapped) == pytest.approx(nsampled)
        sampler_swapped.resample(n=nsampled)
        assert np.sum(sampler_swapped.values()) == pytest.approx(nsampled)
        sampler_swapped.resample(n=nsampled * 3)
        assert np.sum(sampler_swapped.values()) == pytest.approx(nsampled * 3)

    sampler.resample(n=nsampled)
    # start = time.time()
    #     sampler.resample(n=nsampled)
    # print(f"Time taken {(time.time() - start)}")
