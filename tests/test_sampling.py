#  Copyright (c) 2024 zfit
import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import Space, z
import zfit.z.numpy as znp
from zfit.core.space import Limit
from zfit.util.exception import AnalyticSamplingNotImplemented


@pytest.fixture(autouse=True, scope="module")
def setup_teardown_vectors():
    Limit._experimental_allow_vectors = True
    yield
    Limit._experimental_allow_vectors = False


mu_true = 1.5
sigma_true = 1.2
low, high = -3.8, 5.9

obs1 = zfit.Space("obs1", (low, high))


class GaussNoAnalyticSampling(zfit.pdf.Gauss):
    def _analytic_sample(self, n, limits: Space):
        raise AnalyticSamplingNotImplemented  # HACK do make importance sampling work


class UniformNoAnalyticSampling(zfit.pdf.Uniform):
    def _analytic_sample(self, n, limits: Space):
        raise AnalyticSamplingNotImplemented  # HACK do make importance sampling work


def create_gauss1():
    mu = zfit.Parameter("mu_sampling1", mu_true, mu_true - 2.0, mu_true + 7.0)
    sigma = zfit.Parameter(
        "sigma_sampling1", sigma_true, sigma_true - 10.0, sigma_true + 5.0
    )

    gauss_params1 = zfit.pdf.Gauss(
        mu=mu, sigma=sigma, obs=obs1, name="gauss_params1_sampling1"
    )
    return gauss_params1, mu, sigma

def create_gauss_binned1():
    gauss, mu, sigma = create_gauss1()
    return gauss.to_binned(253), mu, sigma


class TmpGaussian(zfit.pdf.BasePDF):
    def __init__(self, obs, mu, sigma, params=None, name: str = "BasePDF", **kwargs):
        params = {"mu": mu, "sigma": sigma}
        super().__init__(obs, params, name=name, **kwargs)

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        mu = self.params["mu"]
        sigma = self.params["sigma"]

        return z.exp((-((x - mu) ** 2)) / (2 * sigma**2))  # non-normalized gaussian


class TmpGaussianPDFNonNormed(zfit.pdf.BasePDF):
    def __init__(self, obs, mu, sigma, params=None, name: str = "BasePDF", **kwargs):
        params = {"mu": mu, "sigma": sigma}
        super().__init__(obs, params, name=name, **kwargs)

    @zfit.supports(norm=True)
    def _pdf(self, x, norm=False):
        x = x.unstack_x()
        mu = self.params["mu"]
        sigma = self.params["sigma"]

        return z.exp((-((x - mu) ** 2)) / (2 * sigma**2))  # non-normalized gaussian


def create_test_gauss1():
    mu2 = zfit.Parameter("mu2_sampling1", mu_true, mu_true - 2.0, mu_true + 7.0)
    sigma2 = zfit.Parameter(
        "sigma2_sampling1", sigma_true, sigma_true - 10.0, sigma_true + 5.0
    )

    test_gauss1 = TmpGaussian(name="test_gauss1", mu=mu2, sigma=sigma2, obs=obs1)
    return test_gauss1, mu2, sigma2

def create_test_gauss_binned1():
    gauss, mu, sigma = create_test_gauss1()
    return gauss.to_binned(273), mu, sigma

def create_test_pdf_overriden_gauss1():
    mu2 = zfit.Parameter("mu2_sampling1", mu_true, mu_true - 2.0, mu_true + 7.0)
    sigma2 = zfit.Parameter(
        "sigma2_sampling1", sigma_true, sigma_true - 10.0, sigma_true + 5.0
    )

    test_gauss1 = TmpGaussianPDFNonNormed(
        name="test_pdf_nonnormed_gauss1", mu=mu2, sigma=sigma2, obs=obs1
    )
    return test_gauss1, mu2, sigma2


gaussian_dists = [create_gauss1, create_test_gauss1]
gaussian_binneddists = [create_gauss_binned1, create_test_gauss_binned1]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 5123, 20000])
def test_mutlidim_sampling(n):
    spaces = [zfit.Space(f"obs{i}", (i * 10, i * 10 + 6)) for i in range(4)]
    pdfs = [
        GaussNoAnalyticSampling(obs=spaces[0], mu=3, sigma=1),
        GaussNoAnalyticSampling(obs=spaces[2], mu=23, sigma=1),
        UniformNoAnalyticSampling(obs=spaces[1], low=12, high=14),
        UniformNoAnalyticSampling(obs=spaces[3], low=32, high=34),
    ]
    prod = zfit.pdf.ProductPDF(pdfs)
    sample = prod.sample(n=n)
    assert sample.value().shape == (n, 4)
    for i, space in enumerate([p.space for p in pdfs]):
        assert all(space.inside(sample.value()[:, i]))


@pytest.mark.flaky(2)  # sampling
@pytest.mark.parametrize("gauss_factory", gaussian_dists)
def test_multiple_limits_sampling(gauss_factory):
    gauss, mu, sigma = gauss_factory()

    low = mu_true - 6 * sigma_true
    high = mu_true + 6 * sigma_true
    low1 = low
    low2 = up1 = (high - low) / 5
    up2 = high
    obs11 = zfit.Space(obs1.obs, (low1, up1))
    obs12 = zfit.Space(obs1.obs, (low2, up2))
    obs_split = obs11 + obs12
    obs = zfit.Space(obs1.obs, (low, high))

    n = 50000
    sample1 = gauss.sample(n=n, limits=obs)
    sample2 = gauss.sample(n=n, limits=obs_split)

    rel_tol = 1e-2
    assert pytest.approx(mu_true, rel_tol) == (np.mean(sample1.value()))
    assert pytest.approx(sigma_true, rel_tol) == (np.std(sample1.value()))
    assert pytest.approx(mu_true, rel_tol) == (np.mean(sample2.value()))
    assert pytest.approx(sigma_true, rel_tol) == (np.std(sample2.value()))


@pytest.mark.parametrize(
    "gauss_factory", gaussian_dists + [create_test_pdf_overriden_gauss1]
)
def test_sampling_fixed(gauss_factory):
    gauss, mu, sigma = gauss_factory()

    n_draws = 1000
    sample_tensor = gauss.create_sampler(n=n_draws, limits=(low, high))
    sample_tensor.resample()
    sampled_from_gauss1 = sample_tensor.value()
    assert max(sampled_from_gauss1[:, 0]) <= high
    assert min(sampled_from_gauss1[:, 0]) >= low
    assert n_draws == len(sampled_from_gauss1[:, 0])

    new_n_draws = 867
    sample_tensor.resample(n=new_n_draws)
    sampled_from_gauss1_small = sample_tensor.value()
    assert new_n_draws == len(sampled_from_gauss1_small[:, 0])
    assert not np.allclose(sampled_from_gauss1[:new_n_draws], sampled_from_gauss1_small)

    gauss_full_sample = gauss.create_sampler(
        n=10000, limits=(mu_true - abs(sigma_true) * 3, mu_true + abs(sigma_true) * 3)
    )
    gauss_full_sample.resample()
    sampled_gauss1_full = gauss_full_sample.value()
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert pytest.approx(mu_true, rel=0.07) == mu_sampled
    assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled

    with mu.set_value(mu_true - 1):
        sample_tensor.resample()
        sampled_from_gauss1 = sample_tensor.value()
        assert max(sampled_from_gauss1[:, 0]) <= high
        assert min(sampled_from_gauss1[:, 0]) >= low
        assert n_draws == len(sampled_from_gauss1[:, 0])

        sampled_gauss1_full = gauss_full_sample.value()
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert pytest.approx(mu_true, rel=0.07) == mu_sampled
        assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled

    gauss_full_sample2 = gauss.create_sampler(n=10000, limits=(-10, 10))

    gauss_full_sample2.resample(params={mu: mu_true - 1.0})
    sampled_gauss2_full = gauss_full_sample2.value()
    mu_sampled = np.mean(sampled_gauss2_full)
    sigma_sampled = np.std(sampled_gauss2_full)
    assert pytest.approx(mu_true - 1.0, rel=0.08) == mu_sampled
    assert pytest.approx(sigma_true, rel=0.08) == sigma_sampled

    gauss_full_sample2.resample(params={sigma: sigma_true + 1.0})
    sampled_gauss2_full = gauss_full_sample2.value()
    mu_sampled = np.mean(sampled_gauss2_full)
    sigma_sampled = np.std(sampled_gauss2_full)
    assert pytest.approx(mu_true, rel=0.07) == mu_sampled
    assert pytest.approx(sigma_true + 1.0, rel=0.07) == sigma_sampled


@pytest.mark.parametrize("gauss_factory", gaussian_dists)
def test_sampling_floating(gauss_factory):
    gauss, mu, sigma = gauss_factory()

    n_draws = 1000
    sampler = gauss.create_sampler(n=n_draws, limits=(low, high))
    sampler.resample()
    sampled_from_gauss1 = sampler
    assert max(sampled_from_gauss1.value()[:, 0]) <= high
    assert min(sampled_from_gauss1.value()[:, 0]) >= low
    assert n_draws == len(sampled_from_gauss1.value()[:, 0])

    nsample = 100000
    gauss_full_sample = gauss.create_sampler(
        n=nsample,
        limits=(mu_true - abs(sigma_true) * 3, mu_true + abs(sigma_true) * 3),
    )

    gauss_full_sample_fixed = gauss.create_sampler(
        n=nsample,
        limits=(mu_true - abs(sigma_true) * 3, mu_true + abs(sigma_true) * 3),
    )
    gauss_full_sample.resample()
    gauss_full_sample_fixed.resample()
    assert gauss_full_sample_fixed.params == {mu.name: mu_true, sigma.name: sigma_true}
    sampled_gauss1_full = gauss_full_sample.value()
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert pytest.approx(mu_true, rel=0.07) == mu_sampled
    assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled

    sampled_gauss1_full_fixed = gauss_full_sample_fixed.value()
    mu_sampled_fixed = np.mean(sampled_gauss1_full_fixed)
    sigma_sampled_fixed = np.std(sampled_gauss1_full_fixed)
    assert pytest.approx(mu_true, rel=0.07) == mu_sampled_fixed
    assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled_fixed

    mu_diff = 0.7
    new_muval = mu_true - mu_diff
    with mu.set_value(new_muval):
        assert mu.value() == new_muval
        sampler.resample()
        sampled_from_gauss1 = sampler.value()
        assert max(sampled_from_gauss1[:, 0]) <= high
        assert min(sampled_from_gauss1[:, 0]) >= low
        assert n_draws == len(sampled_from_gauss1[:, 0])

        sampled_gauss1_full_fixed = gauss_full_sample_fixed.value()
        mu_sampled_fixed = np.mean(sampled_gauss1_full_fixed)
        sigma_sampled_fixed = np.std(sampled_gauss1_full_fixed)
        assert pytest.approx(mu_true, rel=0.07) == mu_sampled_fixed
        assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled_fixed

        gauss_full_sample.resample({mu: new_muval})
        sampled_gauss1_full = gauss_full_sample.value()
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert pytest.approx(new_muval, rel=0.07) == mu_sampled
        assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled


@pytest.mark.parametrize("gauss_binnedfactory", gaussian_binneddists)
def test_binned_sampling_floating(gauss_binnedfactory):
    def std(x, weights):
        return np.sqrt(np.cov(x, aweights=weights))
    gauss, mu, sigma = gauss_binnedfactory()

    n_draws = 10000
    sampler = gauss.create_sampler(n=n_draws, limits=(low, high))
    sampler.resample()
    sampled_from_gauss1 = sampler
    assert n_draws == np.sum(sampled_from_gauss1.counts())

    nsample = 100000
    gauss_full_sample = gauss.create_sampler(
        n=nsample,
    )

    gauss_full_sample_fixed = gauss.create_sampler(
        n=nsample,
    )
    gauss_full_sample.resample()
    gauss_full_sample_fixed.resample()
    assert gauss_full_sample_fixed.params == {mu.name: mu_true, sigma.name: sigma_true}
    sampled_gauss1_full = gauss_full_sample.counts()
    centers = gauss.space.binning.centers[0]
    mu_sampled = np.average(centers, weights=sampled_gauss1_full)
    sigma_sampled = np.sqrt(np.cov(centers, aweights=sampled_gauss1_full))

    assert pytest.approx(mu_true, rel=0.07) == mu_sampled
    assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled

    mu_sampled_fixed = np.average(centers, weights=gauss_full_sample_fixed.counts())
    sigma_sampled_fixed = np.sqrt(np.cov(centers, aweights=gauss_full_sample_fixed.counts()))
    assert pytest.approx(mu_true, rel=0.07) == mu_sampled_fixed
    assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled_fixed

    mu_diff = 0.7
    new_muval = mu_true - mu_diff
    with mu.set_value(new_muval):
        assert mu.value() == new_muval
        sampler.resample()
        assert n_draws == np.sum(sampler.counts())


        mu_sampled_fixed = np.average(centers, weights=sampler.counts())
        sigma_sampled_fixed = std(centers, weights=sampler.counts())
        assert pytest.approx(mu_true, rel=0.07) == mu_sampled_fixed
        assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled_fixed

        gauss_full_sample.resample({mu: new_muval})
        mu_sampled_fixed = np.average(centers, weights=gauss_full_sample.counts())
        sigma_sampled_fixed = std(centers, weights=gauss_full_sample.counts())
        assert pytest.approx(new_muval, rel=0.07) == mu_sampled_fixed
        assert pytest.approx(sigma_true, rel=0.07) == sigma_sampled_fixed


# @pytest.mark.skipif(not zfit.EXPERIMENTAL_FUNCTIONS_RUN_EAGERLY, reason="deadlock in tf.function, issue #35540")  # currently, importance sampling is not working, odd deadlock in TF
@pytest.mark.flaky(3)  # statistical
@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 30000])
def test_importance_sampling(n):
    from zfit.core.sample import accept_reject_sample

    mu_sampler = 5.0
    sigma_sampler = 4.0
    mu_pdf = 4.0
    sigma_pdf = 1.0

    obs_sampler = zfit.Space(obs="obs1", limits=(4.5, 5.5))  # smaller, so pdf is bigger
    obs_pdf = zfit.Space(obs="obs1", limits=(1, 7))

    gauss_sampler = GaussNoAnalyticSampling(
        mu=mu_sampler, sigma=sigma_sampler, obs=obs_sampler
    )
    gauss_pdf = GaussNoAnalyticSampling(mu=mu_pdf, sigma=sigma_pdf, obs=obs_pdf)

    importance_sampling_called = [False]

    class GaussianSampleAndWeights:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @z.function
        def __call__(self, n_to_produce, limits, dtype):
            importance_sampling_called[0] = True
            n_to_produce = znp.asarray(n_to_produce, dtype=tf.int32)
            gaussian_sample = gauss_sampler.sample(
                n=n_to_produce, limits=limits
            ).value()
            weights = gauss_sampler.pdf(gaussian_sample)
            weights_max = None
            thresholds = z.random.uniform(shape=(n_to_produce,), dtype=dtype)
            return gaussian_sample, thresholds, weights, weights_max, n_to_produce

    sample = accept_reject_sample(
        prob=lambda x: gauss_pdf.pdf(x, norm=False), n=n, limits=obs_pdf
    )
    gauss_pdf._sample_and_weights = GaussianSampleAndWeights
    sample2 = gauss_pdf.sample(n=n, limits=obs_pdf)
    assert importance_sampling_called[0]
    assert sample.shape == sample2.value().shape
    assert sample.shape == (n, 1)
    if n > 1000:

        mean = np.mean(sample)
        mean2 = np.mean(sample)
        std = np.std(sample)
        std2 = np.std(sample)
        assert pytest.approx(mu_pdf, rel=0.02) == mean
        assert pytest.approx(mu_pdf, rel=0.02) == mean2
        assert pytest.approx(sigma_pdf, rel=0.02) == std
        assert pytest.approx(sigma_pdf, rel=0.02) == std2


@pytest.mark.flaky(3)  # statistical
def test_importance_sampling_uniform():
    low = -3.0
    high = 7.0
    obs = zfit.Space("obs1", (low, high))
    uniform = UniformNoAnalyticSampling(obs=obs, low=low, high=high)
    importance_sampling_called = [False]

    class GaussianSampleAndWeights:
        def __call__(self, n_to_produce, limits, dtype):
            importance_sampling_called[0] = True

            import tensorflow_probability.python.distributions as tfd

            n_to_produce = znp.asarray(n_to_produce, dtype=tf.int32)
            gaussian = tfd.TruncatedNormal(
                loc=z.constant(-1.0), scale=z.constant(2.0), low=low, high=high
            )
            sample = gaussian.sample(sample_shape=(n_to_produce, 1))
            weights = gaussian.prob(sample)[:, 0]
            thresholds = z.random.uniform(shape=(n_to_produce,), dtype=dtype)
            return sample, thresholds, weights, None, n_to_produce

    uniform._sample_and_weights = GaussianSampleAndWeights
    n_sample = 10000
    sample = uniform.sample(n=n_sample)
    assert importance_sampling_called[0]
    n_bins = 20
    bin_counts, _ = np.histogram(sample.value(), bins=n_bins)
    expected_per_bin = n_sample / n_bins

    assert np.std(bin_counts) < np.sqrt(expected_per_bin) * 2
    assert all(abs(bin_counts - expected_per_bin) < np.sqrt(expected_per_bin) * 5)


def test_sampling_fixed_eventlimits():
    n_samples1 = 500
    n_samples2 = 400  # just to make sure
    n_samples3 = 356  # just to make sure
    n_samples_tot = n_samples1 + n_samples2 + n_samples3

    obs1 = "obs1"
    lower1, upper1 = -10, -9
    lower2, upper2 = 0, 1
    lower3, upper3 = 10, 11
    lower = tf.convert_to_tensor(
        value=tuple(
            [lower1] * n_samples1 + [lower2] * n_samples2 + [lower3] * n_samples3
        )
    )[:, None]
    upper = tf.convert_to_tensor(
        value=tuple(
            [upper1] * n_samples1 + [upper2] * n_samples2 + [upper3] * n_samples3
        )
    )[:, None]

    # limits = zfit.core.sample.EventSpace(obs=obs1, limits=(lower, upper))
    limits = zfit.Space(obs=obs1, limits=(lower, upper))
    gauss1 = GaussNoAnalyticSampling(
        mu=0.3, sigma=4, obs=zfit.Space(obs=obs1, limits=(-12, 12))
    )

    sample = gauss1.sample(n=n_samples_tot, limits=limits).value()
    assert sample.shape[0] == n_samples_tot
    assert all(lower1 <= sample[:n_samples1])
    assert all(sample[:n_samples1] <= upper1)
    assert all(lower2 <= sample[n_samples1:n_samples2])
    assert all(sample[n_samples1:n_samples2] <= upper2)
    assert all(lower3 <= sample[n_samples2:n_samples3])
    assert all(sample[n_samples2:n_samples3] <= upper3)
    # with pytest.raises(InvalidArgumentError):  # cannot use the exact message, () are regex syntax... bug in pytest
    #     _ = gauss1.sample(n=n_samples_tot + 1, limits=limits)  # TODO(Mayou36): catch analytic integral


@pytest.mark.parametrize("n", [1, 2, 3, 5, 13, 30000])
def test_sampling_seed(n):
    data_set = np.random.normal(loc=0.5, scale=0.1, size=100)

    zfit.settings.set_seed(123)

    obs = zfit.Space("x", (0, 1))
    _data_z = zfit.Data.from_numpy(obs=obs, array=data_set)
    gauss = zfit.pdf.Gauss(1, 3, obs=obs)
    gauss_num = GaussNoAnalyticSampling(1, 3, obs=obs)

    sample4 = gauss.sample(n=n)
    sample5 = gauss.sample(n=n)
    sample6 = gauss_num.sample(n=n)
    sample7 = gauss_num.sample(n=n)

    assert sample4.value().shape == (n, 1)
    assert sample5.value().shape == (n, 1)
    assert sample6.value().shape == (n, 1)
    assert sample7.value().shape == (n, 1)
    if n > 4:
        assert not np.allclose(sample4.value(), sample5.value())
        assert not np.allclose(sample6.value(), sample7.value())
        assert not np.allclose(sample4.value(), sample6.value())
        assert not np.allclose(sample5.value(), sample7.value())
