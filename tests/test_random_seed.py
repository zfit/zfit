#  Copyright (c) 2023 zfit
import numpy as np

import zfit


# Set up the simple model for the test
def create_loss(mu, sigma):
    obs = zfit.Space("x", limits=(-10, 10))
    gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    data_np = np.random.normal(mu.value(), sigma.value(), size=10)
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    return zfit.loss.UnbinnedNLL(model=gauss, data=data)


# Test the reproducibility of the fit
def test_reproducibility():
    # Create model and data

    mu = zfit.Parameter("mu", 0.0, -5, 5)
    sigma = zfit.Parameter("sigma", 1.0, 0.1, 10)

    # Set up minimizer
    minimizer = zfit.minimize.Minuit()
    params = [mu, sigma]
    vals = [0.2, 0.7]
    # Perform fit with fixed seed
    seed = 4294
    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed1 = minimizer.minimize(loss)

    # Perform fit again with the same fixed seed
    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed2 = minimizer.minimize(loss)

    # Perform fit without a fixed seed (set to None)
    zfit.settings.set_seed(None)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_no_seed = minimizer.minimize(loss)

    zfit.settings.set_seed(seed)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_fixed_seed3 = minimizer.minimize(loss)

    zfit.settings.set_seed(None)
    loss = create_loss(mu=mu, sigma=sigma)
    with zfit.param.set_values(params, vals):
        result_no_seed2 = minimizer.minimize(loss)

    # Check if the results with fixed seeds are the same
    assert np.allclose(
        result_fixed_seed1.values, result_fixed_seed2.values
    ), "Results with the same fixed seed should be the same"

    # Check if the results without a fixed seed are different
    assert not np.allclose(
        result_fixed_seed1.values, result_no_seed.values
    ), "Results without a fixed seed should be different"

    assert np.allclose(
        result_fixed_seed3.values, result_fixed_seed2.values
    ), "Results with the same fixed seed should be the same"
    assert not np.allclose(
        result_fixed_seed3.values, result_no_seed2.values
    ), "Results without a fixed seed should be different"
    assert not np.allclose(
        result_no_seed.values, result_no_seed2.values
    ), "Results without a fixed seed should be different"
