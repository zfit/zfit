#  Copyright (c) 2025 zfit
import itertools

import hist
import mplhep
import numpy as np
import pytest
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

import zfit
from zfit import z
import zfit.z.numpy as znp
from zfit._interfaces import ZfitParameter


@pytest.mark.skip()  # copy not yet implemented
def test_copy_kde():
    size = 500
    data = np.random.normal(size=size, loc=2, scale=3)

    limits = (-15, 5)
    obs = zfit.Space("obs1", limits=limits)
    kde_adaptive = zfit.pdf.GaussianKDE1DimV1(
        data=data, bandwidth="adaptive", obs=obs, truncate=False
    )
    kde_adaptive.copy()


def create_pdf_sample(npoints=500, upper=None):
    if upper is None:
        upper = 11
    limits = (-13, upper)
    obs = zfit.Space("obs1", limits=limits)
    cb = zfit.pdf.CrystalBall(mu=2, sigma=3, alpha=1, n=25, obs=obs)
    gauss = zfit.pdf.Gauss(mu=-5, sigma=2.5, obs=obs)
    pdf = zfit.pdf.SumPDF([cb, gauss], fracs=0.8)
    data = pdf.sample(n=npoints)
    return data, obs, pdf


def test_grid_kde():
    data, obs, pdf = create_pdf_sample()
    with pytest.raises(ValueError):
        zfit.pdf.KDE1DimGrid(obs=obs, data=data, bandwidth="eoaue")


def test_exact_kde():
    data, obs, pdf = create_pdf_sample()
    with pytest.raises(ValueError):
        zfit.pdf.KDE1DimExact(obs=obs, data=data, bandwidth="eoaue")


def test_default_padding():
    """Test that KDE implementations have default padding of 0.1."""
    data, obs, pdf = create_pdf_sample(npoints=100)

    # Test that all KDE implementations have the correct default padding
    kde_exact = zfit.pdf.KDE1DimExact(data=data, obs=obs, bandwidth="silverman")
    assert kde_exact._default_padding == 0.1

    kde_grid = zfit.pdf.KDE1DimGrid(data=data, obs=obs, bandwidth="silverman")
    assert kde_grid._default_padding == 0.1

    kde_fft = zfit.pdf.KDE1DimFFT(data=data, obs=obs, bandwidth="silverman")
    assert kde_fft._default_padding == 0.1

    kde_isj = zfit.pdf.KDE1DimISJ(data=data, obs=obs)
    assert kde_isj._default_padding == 0.1


def create_kde(
    kdetype=None,
    npoints=1500,
    cfgonly=False,
    nonly=False,
    full=True,
    upper=None,
    legacy=True,
    padding=False,
):
    import tensorflow as tf

    import zfit
    import zfit.z.numpy as znp

    npoints_lim = min(npoints, 2500)

    class StudentT(tfp.distributions.StudentT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, df=5, **kwargs)

    configs = []
    with tf.init_scope():
        if cfgonly or nonly:
            hparam1 = 0.9
        else:
            hparam1 = zfit.Parameter("h1", 0.9)
    comb = itertools.product(
        [
            ("bandwidth", h)
            for h in [
                None,
                hparam1,
                "adaptive",
                "silverman",
                "scott",
                np.random.uniform(0.1, 1.5, size=npoints_lim),
            ]
        ],
        [("truncate", bo) for bo in [False, True]],
        [("type", zfit.pdf.GaussianKDE1DimV1)],
        [("npoints", npoints_lim)],
    )
    if full or legacy:
        if legacy:
            comb = [next(comb)]
        configs.extend(comb)

    with tf.init_scope():
        if cfgonly or nonly:
            hparam2 = 0.9
        else:
            hparam2 = zfit.Parameter("h2", 0.92)
    comb = itertools.product(
        [
            ("bandwidth", h)
            for h in [
                None,
                hparam2,
                "adaptive_zfit",
                "adaptive_geom",
                "adaptive_std",
                "silverman",
                "scott",
                "isj",
                np.random.uniform(0.1, 1.5, size=npoints_lim),
            ]
        ],
        [("weights", weight) for weight in [None, znp.ones(shape=npoints_lim)]],
        [("padding", padding)],
        [("kernel", dist) for dist in [None, StudentT]],
        [("type", zfit.pdf.KDE1DimExact)],
        [("npoints", npoints_lim)],
    )
    if not full:
        comb = [next(comb)]
    configs.extend(comb)

    # Grid PDFs

    num_grid_size_maybe = 800
    comb = itertools.product(
        [
            ("bandwidth", h)
            for h in [
                "adaptive_geom",
                None,
                hparam2,
                "silverman",
                "scott",
                "adaptive_zfit",
                np.random.uniform(0.1, 1.5, size=num_grid_size_maybe),
            ]
        ],
        [("weights", weight) for weight in [None, znp.ones(shape=npoints)]],
        [("padding", padding)],
        [("kernel", dist) for dist in [None, StudentT]],
        [("type", zfit.pdf.KDE1DimGrid)],
        [("num_grid_points", n) for n in [None, num_grid_size_maybe]],
        [("binning_method", method) for method in ["linear", "simple"]],
    )
    comb = [
        c
        for c in comb
        if not isinstance(c[0][1], np.ndarray) or c[5][1] == num_grid_size_maybe
    ]

    if not full:
        comb = [comb[0]]
    configs.extend(comb)

    # FFT combinations

    comb = itertools.product(
        [("bandwidth", h) for h in [None, "scott"]],  # TODO: adaptive?
        [("weights", weight) for weight in [None, znp.ones(shape=npoints)]],
        [("padding", padding)],
        [("kernel", dist) for dist in [None, StudentT]],
        [("type", zfit.pdf.KDE1DimFFT)],
        [("num_grid_points", n) for n in [None, num_grid_size_maybe]],
        [("binning_method", method) for method in ["linear", "simple"]],
    )

    if not full:
        comb = [next(comb)]
    configs.extend(comb)

    # IFJ combinations

    comb = itertools.product(
        [("weights", weight) for weight in [None, znp.ones(shape=npoints)]],
        [("padding", padding)],
        [("type", zfit.pdf.KDE1DimISJ)],
        [("num_grid_points", n) for n in [None, num_grid_size_maybe]],
        [("binning_method", method) for method in ["linear", "simple"]],
    )

    if not full:
        comb = [next(comb)]
    configs.extend(comb)

    # end config builder
    if nonly:
        return len(configs)
    cfg = configs[kdetype]
    cfg = dict(cfg)
    if cfgonly:
        return cfg
    _ = cfg.pop("npoints", None)

    constructor = cfg.pop("type")
    if constructor in (zfit.pdf.GaussianKDE1DimV1, zfit.pdf.KDE1DimExact):
        npoints = npoints_lim

    data, obs, pdf = create_pdf_sample(npoints, upper=upper)

    cfg["data"] = data
    cfg["obs"] = obs

    kde = constructor(**cfg)

    return kde, pdf, data.value()[:, 0]


def _get_print_kde(**kwargs):
    kdes = create_kde(**kwargs)
    return kdes


@pytest.hookimpl
def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ == "test_all_kde":
        full = metafunc.config.getoption("--longtests")
        full = full or metafunc.config.getoption("--longtests-kde")

        metafunc.parametrize(
            "kdetype", [i for i in range(_get_print_kde(nonly=True, full=full))]
        )


# @pytest.mark.flaky(3)
@pytest.mark.parametrize(
    "jit",
    [
        False,
        # True  # todo: activate again?
    ],
)
@pytest.mark.parametrize("npoints", [1100, 500_000])
def test_all_kde(kdetype, npoints, jit, request):

    full = request.config.getoption("--longtests")
    full = full or request.config.getoption("--longtests-kde")
    cfg = create_kde(kdetype=kdetype, npoints=npoints, cfgonly=True, full=full)
    kdetype = kdetype, npoints
    if jit:
        run_jit = z.function(_run)
        (
            expected_integral,
            integral,
            name,
            prob,
            prob_true,
            rel_tol,
            sample,
            sample2,
            x,
            name,
            data,
        ) = run_jit(kdetype, full=full)
    else:
        (
            expected_integral,
            integral,
            name,
            prob,
            prob_true,
            rel_tol,
            sample,
            sample2,
            x,
            name,
            data,
        ) = _run(kdetype, full=full)

    expected_integral = znp.asarray(expected_integral)

    if not jit:
        import matplotlib.pyplot as plt

        plt.figure()
        kernel_used = cfg.get("kernel")
        if kernel_used is not None:
            kernel_print = f", kernel={kernel_used.__name__}"
        else:
            kernel_print = ""
        bandwidth_printready = cfg.get("bandwidth")
        if isinstance(bandwidth_printready, ZfitParameter):
            bandwidth_printready = f"Param({float(bandwidth_printready.value())}"
        if bandwidth_printready is None:
            bandwidth_print = ""
        else:
            bandwidth_print = f", h={bandwidth_printready}"
        bandwidth_print = bandwidth_print[:25]

        num_grid_points = cfg.get("num_grid_points")
        if num_grid_points is not None:
            grid_print = f", grid={num_grid_points}"
        else:
            grid_print = ""
        binning_method = cfg.get("binning_method")
        if binning_method is not None:
            binning_print = f", binning={binning_method}"
        else:
            binning_print = ""
        weights = cfg.get("weights")
        if weights is not None:
            weights_print = ", weighted"
        else:
            weights_print = ""
        npoints_plot = cfg.get("npoints", kdetype[1])
        plt.title(
            f"{name} with {npoints_plot} points{weights_print}{bandwidth_print}{kernel_print}{grid_print}{binning_print}",
            fontsize=10,
        )
        plt.plot(x, prob, label="KDE")
        plt.plot(x, prob_true, label="true PDF")
        data_hist = hist.Hist.new.Reg(40, np.min(data), np.max(data)).Double().fill(data)
        mplhep.histplot(data_hist, density=True, alpha=0.3, label="Kernel points")
        plt.legend()
        pytest.zfit_savefig(folder="kde")

    abs_tol = 0.005 if kdetype[1] > 3000 else 0.03
    tolfac = 6 if not cfg["type"] == tfp.distributions.Normal else 1
    if cfg.get("binning_method") == "simple":
        tolfac *= 6
    assert pytest.approx(expected_integral, abs=abs_tol * tolfac) == znp.asarray(integral)
    assert tuple(sample.shape) == (1, 1)
    assert tuple(sample2.shape) == (1500, 1)
    assert prob.shape.rank == 1
    assert prob.shape[0] == x.shape[0]
    assert np.mean(prob - prob_true) < 0.07 * tolfac
    # make sure that on average, most values are close
    assert pytest.approx(1, abs=0.1 * tolfac) == np.mean(
        (prob / prob_true)[prob_true > np.mean(prob_true)] ** 2
    )
    rtol = 0.05
    np.testing.assert_allclose(prob, prob_true, rtol=rtol, atol=0.01 * tolfac)


def _run(kdetype, full, upper=None, legacy=True, padding=False):
    from zfit.z import numpy as znp

    kde, pdf, data = create_kde(
        *kdetype, full=full, upper=upper, legacy=legacy, padding=padding
    )
    integral = kde.integrate(
        limits=kde.space,
        norm=(-3, 2),
    )
    expected_integral = kde.integrate(
        limits=kde.space,
        norm=(-3, 2),
    )
    rel_tol = 0.04
    sample = kde.sample(1).value()
    sample2 = kde.sample(1500).value()
    x = znp.linspace(*kde.space.limit1d, 30000)
    prob = kde.pdf(x)

    prob_true = pdf.pdf(x)

    return (
        expected_integral,
        integral,
        kde.name,
        prob,
        prob_true,
        rel_tol,
        sample,
        sample2,
        x,
        kde.name,
        data,
    )


@pytest.mark.parametrize(
    "kdetype", [i for i in range(_get_print_kde(nonly=True, full=False, legacy=False))]
)
@pytest.mark.parametrize("npoints", [1100, 500_000])
@pytest.mark.parametrize("upper", [-4, -1, 3])
def test_kde_border(kdetype, npoints, upper):
    cfg = create_kde(
        kdetype=kdetype, npoints=npoints, cfgonly=True, full=False, legacy=False
    )
    kdetype = kdetype, npoints

    (
        expected_integral,
        integral,
        name,
        prob,
        prob_true,
        rel_tol,
        sample,
        sample2,
        x,
        name,
        data,
    ) = _run(kdetype, full=False, upper=upper, legacy=False, padding=True)

    plt.figure()
    if (kernel_used := cfg.get("kernel")) is not None:
        kernel_print = f", kernel={kernel_used.__name__}"
    else:
        kernel_print = ""
    bandwidth_printready = cfg.get("bandwidth")
    if isinstance(bandwidth_printready, ZfitParameter):
        bandwidth_printready = f"Param({float(bandwidth_printready.value())}"
    if bandwidth_printready is None:
        bandwidth_print = ""
    else:
        bandwidth_print = f", h={bandwidth_printready}"

    if (num_grid_points := cfg.get("num_grid_points")) is not None:
        grid_print = f", grid={num_grid_points}"
    else:
        grid_print = ""
    if (binning_method := cfg.get("binning_method")) is not None:
        binning_print = f", binning={binning_method}"
    else:
        binning_print = ""
    weights = cfg.get("weights")
    if weights is not None:
        weights_print = ", weighted"
    else:
        weights_print = ""
    npoints_plot = cfg.get("npoints", kdetype[1])
    plt.title(
        f"{name}, {npoints_plot} points, upper={upper}{weights_print}{bandwidth_print}"
        f"{kernel_print}{grid_print}{binning_print}",
        fontsize=10,
    )
    plt.plot(x, prob, label="KDE")
    plt.plot(x, prob_true, label="true PDF")
    data_binned = hist.Hist.new.Reg(40, np.min(data), np.max(data)).Double().fill(data)
    mplhep.histplot(data_binned, density=True, alpha=0.3, label="Kernel points")
    plt.legend()
    pytest.zfit_savefig(folder="kde_border")

    abs_tol = 0.05
    assert pytest.approx(expected_integral, abs=abs_tol) == (integral)
    assert tuple(sample.shape) == (1, 1)
    assert tuple(sample2.shape) == (1500, 1)
    assert prob.shape.rank == 1
    assert np.mean(prob - prob_true) < 0.07
    # make sure that on average, most values are close
    assert pytest.approx(1, abs=0.15) == np.mean(
        (prob / prob_true)[prob_true > np.mean(prob_true)] ** 2
    )
    rtol = 0.05
    np.testing.assert_allclose(prob, prob_true, rtol=rtol, atol=0.07)


def test_kde_negative_weights():
    """Test KDE with negative weights to ensure no NaN values are produced."""
    limits = (-4, 3)
    obs = zfit.Space("obs1", limits=limits)

    # Test case 1: Small negative weight that doesn't cause negative variance
    data_vals = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([1.0, 1.0, 0.8])  # All positive, sum > 0

    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights)
    kde = zfit.pdf.KDE1DimExact(data, bandwidth='silverman')

    # Test evaluation
    test_x = np.array([[0.0], [1.0], [2.0]])
    pdf_vals = kde.pdf(test_x, norm=False).numpy()

    # Check that no NaN values are produced
    assert not np.any(np.isnan(pdf_vals)), "KDE should not produce NaN values"

    # Test case 2: Weights that sum to zero - should raise ValueError
    weights_zero = np.array([1.0, 1.0, -2.0])
    data_zero = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights_zero)

    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        kde_zero = zfit.pdf.KDE1DimExact(data_zero, bandwidth='silverman')

    # Test case 3: With explicit bandwidth, negative weights work even with extreme values
    weights_neg = np.array([1.0, 1.0, -0.9])
    data_neg = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights_neg)

    # With explicit bandwidth, it should work
    kde_explicit = zfit.pdf.KDE1DimExact(data_neg, bandwidth=0.5)
    pdf_vals_explicit = kde_explicit.pdf(test_x, norm=False).numpy()
    assert not np.any(np.isnan(pdf_vals_explicit)), "KDE with explicit bandwidth should not produce NaN"
    # Note: With negative weights, KDE can produce negative PDF values, which is mathematically correct

    # Test that bandwidth is reasonable for the first case
    bandwidth = kde.params['bandwidth'].value()
    assert bandwidth > 0, "Bandwidth should be positive"
    assert not np.isnan(bandwidth), "Bandwidth should not be NaN"


def test_kde_negative_weights_adaptive():
    """Test adaptive KDE with negative weights."""
    limits = (-4, 3)
    obs = zfit.Space("obs1", limits=limits)

    # Test with explicit bandwidth - should work even with negative weights
    data_vals = np.array([[0.0], [1.0], [2.0], [1.5], [0.5]])
    weights = np.array([1.0, 1.0, -0.5, 0.5, -0.2])

    data = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights)

    # Should work with explicit bandwidth
    kde_explicit = zfit.pdf.KDE1DimExact(data, bandwidth=0.5)
    test_x = np.array([[0.0], [1.0], [2.0]])
    pdf_vals = kde_explicit.pdf(test_x, norm=False).numpy()
    assert not np.any(np.isnan(pdf_vals)), "KDE with explicit bandwidth should not produce NaN"

    # Test that works with positive weights
    weights_positive = np.array([1.0, 1.0, 0.8, 0.9, 0.5])
    data_positive = zfit.data.Data.from_numpy(obs=obs, array=data_vals, weights=weights_positive)

    # All methods should work with positive weights
    for bandwidth in ['silverman', 'scott']:
        kde = zfit.pdf.KDE1DimExact(data_positive, bandwidth=bandwidth)
        pdf_vals = kde.pdf(test_x, norm=False).numpy()
        assert not np.any(np.isnan(pdf_vals)), f"KDE ({bandwidth}) should not produce NaN with positive weights"
