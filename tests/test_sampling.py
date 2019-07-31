#  Copyright (c) 2019 zfit

import numpy as np
import pytest
import tensorflow as tf

import zfit
from zfit import ztf
from zfit.core.sample import accept_reject_sample
from zfit.util.execution import SessionHolderMixin
from zfit.core.testing import setup_function, teardown_function, tester

mu_true = 1.5
sigma_true = 1.2
low, high = -3.8, 2.9

obs1 = 'obs1'


def create_gauss1():
    mu = zfit.Parameter("mu_sampling1", mu_true, mu_true - 2., mu_true + 7.)
    sigma = zfit.Parameter("sigma_sampling1", sigma_true, sigma_true - 10., sigma_true + 5.)

    gauss_params1 = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs1, name="gauss_params1_sampling1")
    return gauss_params1, mu, sigma


class TestGaussian(zfit.pdf.BasePDF):

    def __init__(self, obs, mu, sigma, params=None,
                 name: str = "BasePDF", **kwargs):
        params = {'mu': mu, 'sigma': sigma}
        super().__init__(obs, params, name=name, **kwargs)

    def _unnormalized_pdf(self, x, norm_range=False):
        x = x.unstack_x()
        mu = self.params['mu']
        sigma = self.params['sigma']

        return ztf.exp((-(x - mu) ** 2) / (
            2 * sigma ** 2))  # non-normalized gaussian


def create_test_gauss1():
    mu2 = zfit.Parameter("mu2_sampling1", mu_true, mu_true - 2., mu_true + 7.)
    sigma2 = zfit.Parameter("sigma2_sampling1", sigma_true, sigma_true - 10., sigma_true + 5.)

    test_gauss1 = TestGaussian(name="test_gauss1", mu=mu2, sigma=sigma2, obs=obs1)
    return test_gauss1, mu2, sigma2


gaussian_dists = [lambda: create_gauss1(), lambda: create_test_gauss1()]


@pytest.mark.parametrize('gauss_factory', gaussian_dists)
def test_sampling_fixed(gauss_factory):
    gauss, mu, sigma = gauss_factory()
    import tensorflow as tf
    n_draws = 1000
    n_draws_param = tf.Variable(initial_value=n_draws, trainable=False, dtype=tf.int64,
                                name='n_draws',
                                use_resource=True)  # variable to have something changeable, predictable
    zfit.run(n_draws_param.initializer)
    sample_tensor = gauss.create_sampler(n=n_draws_param, limits=(low, high))
    sample_tensor.resample()
    sampled_from_gauss1 = zfit.run(sample_tensor)
    assert max(sampled_from_gauss1[:, 0]) <= high
    assert min(sampled_from_gauss1[:, 0]) >= low
    assert n_draws == len(sampled_from_gauss1[:, 0])

    new_n_draws = 867
    n_draws_param.load(new_n_draws, session=zfit.run.sess)
    sample_tensor.resample()
    sampled_from_gauss1_small = zfit.run(sample_tensor)
    assert new_n_draws == len(sampled_from_gauss1_small[:, 0])
    assert not np.allclose(sampled_from_gauss1[:new_n_draws], sampled_from_gauss1_small)
    n_draws_param.load(n_draws, session=zfit.run.sess)

    gauss_full_sample = gauss.create_sampler(n=10000,
                                             limits=(mu_true - abs(sigma_true) * 3, mu_true + abs(sigma_true) * 3))
    gauss_full_sample.resample()
    sampled_gauss1_full = zfit.run(gauss_full_sample)
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    with mu.set_value(mu_true - 1):
        sample_tensor.resample()
        sampled_from_gauss1 = zfit.run(sample_tensor)
        assert max(sampled_from_gauss1[:, 0]) <= high
        assert min(sampled_from_gauss1[:, 0]) >= low
        assert n_draws == len(sampled_from_gauss1[:, 0])

        sampled_gauss1_full = zfit.run(gauss_full_sample)
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert mu_sampled == pytest.approx(mu_true, rel=0.07)
        assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    gauss_full_sample2 = gauss.create_sampler(n=10000, limits=(-10, 10))

    gauss_full_sample2.resample(param_values={mu: mu_true-1.0})
    sampled_gauss2_full = zfit.run(gauss_full_sample2)
    mu_sampled = np.mean(sampled_gauss2_full)
    sigma_sampled = np.std(sampled_gauss2_full)
    assert mu_sampled == pytest.approx(mu_true-1.0, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    gauss_full_sample2.resample(param_values={sigma: sigma_true+1.0})
    sampled_gauss2_full = zfit.run(gauss_full_sample2)
    mu_sampled = np.mean(sampled_gauss2_full)
    sigma_sampled = np.std(sampled_gauss2_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true+1.0, rel=0.07)


@pytest.mark.parametrize('gauss_factory', gaussian_dists)
def test_sampling_floating(gauss_factory):
    gauss, mu, sigma = gauss_factory()

    n_draws = 1000
    sampler = gauss.create_sampler(n=n_draws, limits=(low, high), fixed_params=False)
    sampler.resample()
    sampled_from_gauss1 = zfit.run(sampler)
    assert max(sampled_from_gauss1[:, 0]) <= high
    assert min(sampled_from_gauss1[:, 0]) >= low
    assert n_draws == len(sampled_from_gauss1[:, 0])

    gauss_full_sample = gauss.create_sampler(n=10000,
                                             limits=(mu_true - abs(sigma_true) * 3, mu_true + abs(sigma_true) * 3),
                                             fixed_params=False)
    gauss_full_sample.resample()
    sampled_gauss1_full = zfit.run(gauss_full_sample)
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    mu_diff = 0.7
    with mu.set_value(mu_true - mu_diff):
        assert zfit.run(mu) == mu_true - mu_diff
        sampler.resample()
        sampled_from_gauss1 = zfit.run(sampler)
        assert max(sampled_from_gauss1[:, 0]) <= high
        assert min(sampled_from_gauss1[:, 0]) >= low
        assert n_draws == len(sampled_from_gauss1[:, 0])

        gauss_full_sample.resample()
        sampled_gauss1_full = zfit.run(gauss_full_sample)
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert mu_sampled == pytest.approx(mu_true - mu_diff, rel=0.07)
        assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)


@pytest.mark.flaky(3)  # statistical
def test_importance_sampling():
    mu_sampler = 5.
    sigma_sampler = 4.
    mu_pdf = 4.
    sigma_pdf = 1.

    obs_sampler = zfit.Space(obs='obs1', limits=(4.5, 5.5))  # smaller, so pdf is bigger
    obs_pdf = zfit.Space(obs='obs1', limits=(1, 7))
    gauss_sampler = zfit.pdf.Gauss(mu=mu_sampler, sigma=sigma_sampler, obs=obs_sampler)
    gauss_pdf = zfit.pdf.Gauss(mu=mu_pdf, sigma=sigma_pdf, obs=obs_pdf)

    importance_sampling_called = [False]

    class GaussianSampleAndWeights(SessionHolderMixin):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.n_to_produce = tf.Variable(initial_value=-42, dtype=tf.int64, use_resource=True,
            #                                 trainable=False, validate_shape=False)
            # self.sess.run(self.n_to_produce.initializer)
            # self.dtype = dtype
            # self.limits = limits

        def __call__(self, n_to_produce, limits, dtype):
            importance_sampling_called[0] = True
            n_to_produce = tf.cast(n_to_produce, dtype=tf.int32)
            # assign_op = self.n_to_produce.assign(n_to_produce)
            # with tf.control_dependencies([assign_op]):
            gaussian_sample = gauss_sampler._create_sampler_tensor(n=n_to_produce, limits=limits,
                                                                   fixed_params=False, name='asdf')[2]
            weights = gauss_sampler.pdf(gaussian_sample)
            weights_max = None
            thresholds = tf.random_uniform(shape=(n_to_produce,), dtype=dtype)
            return gaussian_sample, thresholds, weights, weights_max, n_to_produce

    sample = accept_reject_sample(prob=gauss_pdf.unnormalized_pdf, n=30000, limits=obs_pdf)
    gauss_pdf._sample_and_weights = GaussianSampleAndWeights
    sample2 = gauss_pdf.sample(n=30000, limits=obs_pdf)
    assert importance_sampling_called[0]
    sample_np, sample_np2 = zfit.run([sample, sample2])

    mean = np.mean(sample_np)
    mean2 = np.mean(sample_np2)
    std = np.std(sample_np)
    std2 = np.std(sample_np2)
    assert mean == pytest.approx(mu_pdf, rel=0.02)
    assert mean2 == pytest.approx(mu_pdf, rel=0.02)
    assert std == pytest.approx(sigma_pdf, rel=0.02)
    assert std2 == pytest.approx(sigma_pdf, rel=0.02)


@pytest.mark.flaky(3)  # statistical
def test_importance_sampling_uniform():
    low = -3.
    high = 7.
    obs = zfit.Space("obs1", (low, high))
    uniform = zfit.pdf.Uniform(obs=obs, low=low, high=high)
    importance_sampling_called = [False]

    class GaussianSampleAndWeights(SessionHolderMixin):

        def __call__(self, n_to_produce, limits, dtype):
            importance_sampling_called[0] = True

            import tensorflow_probability.python.distributions as tfd
            n_to_produce = tf.cast(n_to_produce, dtype=tf.int32)
            gaussian = tfd.TruncatedNormal(loc=ztf.constant(-1.), scale=ztf.constant(2.),
                                           low=low, high=high)
            sample = gaussian.sample(sample_shape=(n_to_produce, 1))
            weights = gaussian.prob(sample)[:, 0]
            thresholds = tf.random_uniform(shape=(n_to_produce,), dtype=dtype)
            return sample, thresholds, weights, None, n_to_produce

    uniform._sample_and_weights = GaussianSampleAndWeights
    n_sample = 10000
    sample = uniform.sample(n=n_sample)
    assert importance_sampling_called[0]
    sample_np = zfit.run(sample)
    n_bins = 20
    bin_counts, _ = np.histogram(sample_np, bins=n_bins)
    expected_per_bin = n_sample / n_bins

    assert np.std(bin_counts) < np.sqrt(expected_per_bin) * 2
    assert all(abs(bin_counts - expected_per_bin) < np.sqrt(expected_per_bin) * 5)
    # import matplotlib.pyplot as plt
    # plt.hist(sample_np, bins=40)
    # plt.show()

def test_sampling_fixed_eventlimits():
    n_samples1 = 500
    n_samples2 = 400  # just to make sure
    n_samples3 = 356  # just to make sure
    n_samples_tot = n_samples1 + n_samples2 + n_samples3

    obs1 = "obs1"
    lower1, upper1 = -10, -9
    lower2, upper2 = 0, 1
    lower3, upper3 = 10, 11
    lower = tf.convert_to_tensor(tuple([lower1] * n_samples1 + [lower2] * n_samples2 + [lower3] * n_samples3))
    upper = tf.convert_to_tensor(tuple([upper1] * n_samples1 + [upper2] * n_samples2 + [upper3] * n_samples3))
    lower = ((lower,),)
    upper = ((upper,),)
    limits = zfit.core.sample.EventSpace(obs=obs1, limits=(lower, upper))
    gauss1 = zfit.pdf.Gauss(mu=0.3, sigma=4, obs=zfit.Space(obs=obs1, limits=(-7, 8)))

    sample = gauss1.sample(n=n_samples_tot, limits=limits)
    sample_np = zfit.run(sample)
    assert sample_np.shape[0] == n_samples_tot
    assert all(lower1 <= sample_np[:n_samples1])
    assert all(sample_np[:n_samples1] <= upper1)
    assert all(lower2 <= sample_np[n_samples1:n_samples2])
    assert all(sample_np[n_samples1:n_samples2] <= upper2)
    assert all(lower3 <= sample_np[n_samples2:n_samples3])
    assert all(sample_np[n_samples2:n_samples3] <= upper3)
    with pytest.raises(ValueError,
                       match="are incompatible"):  # cannot use the exact message, () are regex syntax... bug in pytest
        _ = gauss1.sample(n=n_samples_tot + 1, limits=limits)
