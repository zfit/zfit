#  Copyright (c) 2024 zfit
import pytest
import numpy as np
import zfit
import zfit.z.numpy as znp


@pytest.mark.parametrize(
    ("mu_true", "sigma_true", "lambd_true", "obs_limits_low", "obs_limits_high"),
    [
        (2, 1, 0.5, -15, 30),
        (-2, 0.5, 0.1, -15, 90),
        (-5, 3, 1, -25, 15),
        (10_000, 1, 1, 9_070, 10_030), # Large positive mu
        (-10_000, 1, 1, -10_030, -9_070), # Large negative mu
        (0, 0.0001, 1, -10, 10), # Test small sigma value, distribution ~= exponential in this case
        (0, 0.000001, 1, -10, 10), # Test small sigma value, distribution ~= exponential in this case
        (0, 1, 0.001, -10, 10_000), # Test small rate = very large right tail
        (0, 1, 0.0001, -10, 100_000), # Test small rate = very large right tail
        (1, 1, 1_000, -10, 15), # Test large lambd value, distribution =~ gaussian in this case
        (1, 1, 100_000, -10, 15), # Test large lambd value, distribution =~ gaussian in this case
        (1, 1, 10_000_000, -10, 15), # Test large lambd value, distribution =~ gaussian in this case
    ]
)
def test_expmodgauss(mu_true, sigma_true, lambd_true, obs_limits_low, obs_limits_high):
    """
    Parameterized test for the ExpModGauss PDF.

    Parameterize also the obs_limits_low, obs_limits_high because this may be a
    long tailed distribution for small lambd values. I am testing a sampled mean
    and stdev against the theoretical infinite extend values for the distribution.
    The obs_limits_low, obs_limits_high should be large enough to encompass the
    whole shape of the distribution
    """
    mu = zfit.Parameter("mu", mu_true)
    sigma = zfit.Parameter("sigma", sigma_true)
    lambd = zfit.Parameter("lambda", lambd_true)

    obs = zfit.Space(obs="obs", limits=(obs_limits_low, obs_limits_high))
    expmodgauss = zfit.pdf.ExpModGauss(mu=mu, sigma=sigma, lambd=lambd, obs=obs)

    samples = expmodgauss.sample(100_000)["obs"]
    probabilities = expmodgauss.pdf(x=samples)

    assert znp.all(np.isfinite(probabilities))
    assert znp.all(probabilities>=0)
    assert pytest.approx(mu_true + 1 / lambd_true, rel=5e-2) == znp.mean(samples)
    assert pytest.approx(znp.sqrt(sigma_true ** 2 + 1 / lambd_true ** 2), rel=5e-2) == znp.std(samples)
