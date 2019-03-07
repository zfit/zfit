import numpy as np
import pytest

import zfit
from zfit import ztf
from zfit.core.sample import accept_reject_sample

mu_true = 1.5
sigma_true = 1.2
low, high = -4.1, 2.9
mu = zfit.Parameter("mu_sampling1", mu_true, mu_true - 2., mu_true + 7.)
sigma = zfit.Parameter("sigma_sampling1", sigma_true, sigma_true - 10., sigma_true + 5.)

obs1 = 'obs1'

gauss_params1 = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs1, name="gauss_params1_sampling1")


class TestGaussian(zfit.core.basepdf.BasePDF):

    def _unnormalized_pdf(self, x, norm_range=False):
        x = x.unstack_x()
        return ztf.exp((-(x - mu) ** 2) / (2 * sigma ** 2))  # non-normalized gaussian


mu2 = zfit.Parameter("mu2_sampling1", mu_true, mu_true - 2., mu_true + 7.)
sigma2 = zfit.Parameter("sigma2_sampling1", sigma_true, sigma_true - 10., sigma_true + 5.)

test_gauss1 = TestGaussian(name="test_gauss1", obs=obs1)

gaussian_dists = [test_gauss1, gauss_params1]


@pytest.mark.parametrize('gauss', gaussian_dists)
def test_sampling_fixed(gauss):
    import tensorflow as tf
    n_draws = 1000
    n_draws_param = tf.Variable(initial_value=n_draws, trainable=False, dtype=tf.int64,
                                name='n_draws',
                                use_resource=True)  # just variable to have something changeable, predictable
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
    n_draws_param.load(n_draws, session=zfit.run.sess)

    gauss_full_sample = gauss.create_sampler(n=10000,
                                             limits=(mu_true - abs(sigma_true) * 5, mu_true + abs(sigma_true) * 5))
    gauss_full_sample.resample()
    sampled_gauss1_full = zfit.run(gauss_full_sample)
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    with mu.set_value(mu_true - 1), mu2.set_value(mu_true - 1):
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


@pytest.mark.parametrize('gauss', gaussian_dists)
def test_sampling_floating(gauss):
    n_draws = 1000
    sampler = gauss.create_sampler(n=n_draws, limits=(low, high), fixed_params=False)
    sampler.resample()
    sampled_from_gauss1 = zfit.run(sampler)
    assert max(sampled_from_gauss1[:, 0]) <= high
    assert min(sampled_from_gauss1[:, 0]) >= low
    assert n_draws == len(sampled_from_gauss1[:, 0])

    gauss_full_sample = gauss.create_sampler(n=10000,
                                             limits=(mu_true - abs(sigma_true) * 5, mu_true + abs(sigma_true) * 5),
                                             fixed_params=False)
    gauss_full_sample.resample()
    sampled_gauss1_full = zfit.run(gauss_full_sample)
    mu_sampled = np.mean(sampled_gauss1_full)
    sigma_sampled = np.std(sampled_gauss1_full)
    assert mu_sampled == pytest.approx(mu_true, rel=0.07)
    assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)

    with mu.set_value(mu_true - 1), mu2.set_value(mu_true - 1):
        assert zfit.run(mu) == mu_true - 1
        assert zfit.run(mu2) == mu_true - 1
        sampler.resample()
        sampled_from_gauss1 = zfit.run(sampler)
        assert max(sampled_from_gauss1[:, 0]) <= high
        assert min(sampled_from_gauss1[:, 0]) >= low
        assert n_draws == len(sampled_from_gauss1[:, 0])

        gauss_full_sample.resample()
        sampled_gauss1_full = zfit.run(gauss_full_sample)
        mu_sampled = np.mean(sampled_gauss1_full)
        sigma_sampled = np.std(sampled_gauss1_full)
        assert mu_sampled == pytest.approx(mu_true - 1, rel=0.07)
        assert sigma_sampled == pytest.approx(sigma_true, rel=0.07)


@pytest.mark.flaky(3)  # statistical
def test_importance_sampling():
    import tensorflow as tf

    mu_sampler = 5.
    sigma_sampler = 4.
    mu_pdf = 4.
    sigma_pdf = 1.

    obs_sampler = zfit.Space(obs='obs1', limits=(4.5, 5.5))  # smaller, so pdf is bigger
    obs_pdf = zfit.Space(obs='obs1', limits=(1, 9))
    gauss_sampler = zfit.pdf.Gauss(mu=mu_sampler, sigma=sigma_sampler, obs=obs_sampler)
    gauss_pdf = zfit.pdf.Gauss(mu=mu_pdf, sigma=sigma_pdf, obs=obs_pdf)

    def gaussian_sample_and_weights(n_to_produce, limits, dtype):
        gaussian_sample = gauss_sampler.sample(n=n_to_produce, limits=limits)
        weights = gauss_sampler.pdf(gaussian_sample)
        weights_max = tf.reduce_max(weights) * 0.7
        thresholds = tf.random_uniform(shape=(n_to_produce,))
        return gaussian_sample, thresholds, weights, weights_max, n_to_produce

    sample = accept_reject_sample(prob=gauss_pdf.unnormalized_pdf, n=30000, limits=obs_pdf)
    gauss_pdf._sample_and_weights = gaussian_sample_and_weights
    sample2 = gauss_pdf.sample(n=30000, limits=obs_pdf)
    sample_np, sample_np2 = zfit.run([sample, sample2])

    mean = np.mean(sample_np)
    mean2 = np.mean(sample_np2)
    std = np.std(sample_np)
    std2 = np.std(sample_np2)
    assert mean == pytest.approx(mu_pdf, rel=0.01)
    assert mean2 == pytest.approx(mu_pdf, rel=0.01)
    assert std == pytest.approx(sigma_pdf, rel=0.01)
    assert std2 == pytest.approx(sigma_pdf, rel=0.01)
